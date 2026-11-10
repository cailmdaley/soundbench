//! Print chimera_probe(f, x, y) as hex bits for the (f, x, y) triples read from stdin (hex bits).
use std::io::BufRead;
fn main() {
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let mut it = line.split_whitespace();
        let f: u32 = it.next().unwrap().parse().unwrap();
        let x = f64::from_bits(u64::from_str_radix(it.next().unwrap(), 16).unwrap());
        let y = f64::from_bits(u64::from_str_radix(it.next().unwrap(), 16).unwrap());
        println!("{:016x}", chimera_wasm::chimera_probe(f, x, y).to_bits());
    }
}
