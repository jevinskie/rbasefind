extern crate byteorder;
extern crate clap;
extern crate fnv;
extern crate num_cpus;
extern crate ocl;
extern crate pbr;
extern crate regex;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use clap::Parser;
use fnv::FnvHashSet;
use ocl::{Buffer, Context, Device, Kernel, MemFlags, Program, Queue, SpatialDims};
use pbr::MultiBar;
use regex::bytes::Regex;
use std::collections::BinaryHeap;
use std::error::Error;
use std::fs::File;
use std::io::Cursor;
use std::io::prelude::*;
use std::sync::Arc;
use std::thread;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Scan a flat 32-bit binary and attempt to brute-force the base address via string/pointer comparison. Based on the excellent basefind.py by mncoppola."
)]
pub struct Config {
    #[arg(short, long)]
    big_endian: bool,
    #[arg(long, default_value_t = 10)]
    min_str_len: usize,
    #[arg(long, default_value_t = 10)]
    max_matches: usize,
    #[arg(short, long, default_value_t = 0x1000)]
    offset: u32,
    #[arg(short, long, default_value_t = num_cpus::get())]
    threads: usize,
    #[arg(short = 'c', long)]
    opencl: bool,
    filename: String,
}

pub struct Interval {
    start_addr: u32,
    end_addr: u32,
}

impl Interval {
    fn get_range(
        index: usize,
        max_threads: usize,
        offset: u32,
    ) -> Result<Interval, Box<dyn Error + Send + Sync>> {
        if index >= max_threads {
            return Err("Invalid index specified".into());
        }

        if offset.count_ones() != 1 {
            return Err("Invalid additive offset".into());
        }

        let mut start_addr = index as u64 * u64::from(u32::MAX).div_ceil(max_threads as u64);
        let mut end_addr = (index as u64 + 1) * u64::from(u32::MAX).div_ceil(max_threads as u64);

        // Mask the address such that it's aligned to the 2^N offset.
        start_addr &= !(u64::from(offset) - 1);
        if end_addr >= u64::from(u32::MAX) {
            end_addr = u64::from(u32::MAX);
        } else {
            end_addr &= !(u64::from(offset) - 1);
        }

        let interval = Interval {
            start_addr: start_addr as u32,
            end_addr: end_addr as u32,
        };

        Ok(interval)
    }
}

fn get_strings(config: &Config, buffer: &[u8]) -> Result<Vec<u32>, Box<dyn Error>> {
    let mut strings = Vec::<u32>::new();

    let reg_str = format!("[ -~\\t\\r\\n]{{{},}}\x00", config.min_str_len);
    for mat in Regex::new(&reg_str)?.find_iter(buffer) {
        strings.push(mat.start() as u32);
    }

    Ok(strings)
}

fn get_pointers(config: &Config, buffer: &[u8]) -> Result<FnvHashSet<u32>, Box<dyn Error>> {
    let mut pointers = FnvHashSet::default();
    let mut rdr = Cursor::new(&buffer);
    loop {
        let res = if config.big_endian {
            rdr.read_u32::<BigEndian>()
        } else {
            rdr.read_u32::<LittleEndian>()
        };
        match res {
            Ok(v) => pointers.insert(v),
            Err(_) => break,
        };
    }

    Ok(pointers)
}

fn find_matches(
    config: &Config,
    strings: &Vec<u32>,
    pointers: &FnvHashSet<u32>,
    scan_interval: usize,
    mut pb: pbr::ProgressBar<pbr::Pipe>,
) -> Result<BinaryHeap<(usize, u32)>, Box<dyn Error + Send + Sync>> {
    let interval = Interval::get_range(scan_interval, config.threads, config.offset)?;
    let mut current_addr = interval.start_addr;
    let mut heap = BinaryHeap::<(usize, u32)>::new();
    pb.total = ((interval.end_addr - interval.start_addr) / config.offset) as u64;
    while current_addr <= interval.end_addr {
        let mut news = FnvHashSet::default();
        for s in strings {
            match s.checked_add(current_addr) {
                Some(add) => news.insert(add),
                // strings are now ordered, if it overflows u32 we can just break instead of continuing,
                // it will overflow u32 also for the strings after.
                None => break,
            };
        }
        let intersection: FnvHashSet<_> = news.intersection(pointers).collect();
        if !intersection.is_empty() {
            heap.push((intersection.len(), current_addr));
        }
        match current_addr.checked_add(config.offset) {
            Some(_) => current_addr += config.offset,
            None => break,
        };
        pb.inc();
    }
    pb.finish();

    Ok(heap)
}

fn cpu_search(
    config: &Arc<Config>,
    strings: &Arc<Vec<u32>>,
    pointers: &Arc<FnvHashSet<u32>>,
) -> BinaryHeap<(usize, u32)> {
    let mut children = vec![];

    let mb = MultiBar::new();
    mb.println(&format!("Scanning with {} threads...", config.threads));
    for i in 0..config.threads {
        let mut pb = mb.create_bar(100);
        pb.show_message = true;
        let child_config = Arc::clone(config);
        let child_strings = Arc::clone(strings);
        let child_pointers = Arc::clone(pointers);
        children.push(thread::spawn(move || {
            find_matches(&child_config, &child_strings, &child_pointers, i, pb)
        }));
    }

    mb.listen();

    // Merge all of the heaps.
    let mut heap = BinaryHeap::<(usize, u32)>::new();
    for child in children {
        heap.append(&mut child.join().unwrap().unwrap());
    }

    heap
}

fn opencl_search(
    config: &Arc<Config>,
    strings: &[u32],
    pointers: &[u32],
) -> BinaryHeap<(usize, u32)> {
    let compute_program = r#"
        __kernel void find(__global uint* strings, 
        ulong str_count, 
        __global uint* pointers,
        ulong ptr_count,
        __global uint* results) {
            uint current_addr = get_global_id(0) * 0x1000;
            uint intersect_count = 0;
            for (uint i=0; i<str_count; i++) {
                unsigned long translated_string = ((ulong)strings[i]) + ((ulong)current_addr);
                if (translated_string > 0xffffffff) {
                        break;
                }
                for (uint j=0; j<ptr_count; j++) {
                    if (pointers[j] == translated_string) {
                        intersect_count += 1;
                    }
                }
            }
            results[get_global_id(0)] = intersect_count;
        }
    "#;
    if config.offset != 0x1000 {
        panic!("in opencl mode we support only 0x1000 offset");
    }
    let context = Context::builder()
        .devices(
            Device::specifier()
                .type_flags(ocl::flags::DEVICE_TYPE_GPU)
                .first(),
        )
        .build()
        .unwrap();

    let device = context.devices()[0];
    let queue = Queue::new(&context, device, None).unwrap();
    let program = Program::builder()
        .src(compute_program)
        .devices(device)
        .build(&context)
        .unwrap();

    let string_buffer = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_only())
        .len(strings.len())
        .copy_host_slice(strings)
        .build()
        .expect("cannot build the strings buffer");

    let pointer_buffer = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().read_only())
        .len(pointers.len())
        .copy_host_slice(pointers)
        .build()
        .expect("cannot build the pointers buffer");

    let result_buffer = Buffer::<u32>::builder()
        .queue(queue.clone())
        .flags(MemFlags::new().write_only())
        .len(0x100000)
        .build()
        .expect("cannot build the results buffer");

    let kernel = Kernel::builder()
        .program(&program)
        .name("find")
        .queue(queue.clone())
        .global_work_size(SpatialDims::One(0x100000))
        .arg_named("strings", &string_buffer)
        .arg_named("str_count", string_buffer.len())
        .arg_named("pointers", &pointer_buffer)
        .arg_named("ptr_count", pointer_buffer.len())
        .arg_named("results", &result_buffer)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut vec_result = vec![0u32; result_buffer.len()];
    result_buffer.read(&mut vec_result).enq().unwrap();

    queue.finish().unwrap();
    let mut heap = BinaryHeap::<(usize, u32)>::new();
    for (page, count) in vec_result.iter().enumerate().take(0x100000) {
        if *count > 0 {
            heap.push((*count as usize, (page * 0x1000) as u32));
        }
    }
    heap
}

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    // Read in the input file. We jam it all into memory for now.
    let mut f = File::open(&config.filename)?;
    let mut buffer = Vec::new();
    f.read_to_end(&mut buffer)?;

    // Find indices of strings.
    let strings = get_strings(&config, &buffer)?;

    if strings.is_empty() {
        return Err("No strings found in target binary".into());
    }
    eprintln!("Located {} strings", strings.len());

    let pointers = get_pointers(&config, &buffer)?;
    eprintln!("Located {} pointers", pointers.len());

    let shared_config = Arc::new(config);

    let mut heap = if shared_config.opencl {
        let mut pointers_vec = Vec::<u32>::new();
        for p in pointers {
            pointers_vec.push(p);
        }
        opencl_search(&shared_config, &strings, &pointers_vec)
    } else {
        let shared_strings = Arc::new(strings);
        let shared_pointers = Arc::new(pointers);
        cpu_search(&shared_config, &shared_strings, &shared_pointers)
    };

    // Print (up to) top N results.
    for _ in 0..shared_config.max_matches {
        let (count, addr) = match heap.pop() {
            Some(v) => v,
            None => break,
        };
        println!("0x{:08x}: {}", addr, count);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn find_matches_invalid_interval() {
        let _ = Interval::get_range(1, 1, 0x1000).unwrap();
    }

    #[test]
    fn find_matches_single_cpu_interval_0() {
        let interval = Interval::get_range(0, 1, 0x1000).unwrap();
        assert_eq!(interval.start_addr, u32::MIN);
        assert_eq!(interval.end_addr, u32::MAX);
    }

    #[test]
    fn find_matches_double_cpu_interval_0() {
        let interval = Interval::get_range(0, 2, 0x1000).unwrap();
        assert_eq!(interval.start_addr, u32::MIN);
        assert_eq!(interval.end_addr, 0x80000000);
    }

    #[test]
    fn find_matches_double_cpu_interval_1() {
        let interval = Interval::get_range(1, 2, 0x1000).unwrap();
        assert_eq!(interval.start_addr, 0x80000000);
        assert_eq!(interval.end_addr, u32::MAX);
    }

    #[test]
    fn find_matches_triple_cpu_interval_0() {
        let interval = Interval::get_range(0, 3, 0x1000).unwrap();
        assert_eq!(interval.start_addr, u32::MIN);
        assert_eq!(interval.end_addr, 0x55555000);
    }

    #[test]
    fn find_matches_triple_cpu_interval_1() {
        let interval = Interval::get_range(1, 3, 0x1000).unwrap();
        assert_eq!(interval.start_addr, 0x55555000);
        assert_eq!(interval.end_addr, 0xAAAAA000);
    }

    #[test]
    fn find_matches_triple_cpu_interval_2() {
        let interval = Interval::get_range(2, 3, 0x1000).unwrap();
        assert_eq!(interval.start_addr, 0xAAAAA000);
        assert_eq!(interval.end_addr, u32::MAX);
    }
}
