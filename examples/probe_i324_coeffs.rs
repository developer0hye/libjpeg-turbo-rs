//! Issue #324 debug probe: compare the decoded coefficient arrays of two JPEGs.
//!
//! Tells apart the two possible causes of a byte divergence: different
//! coefficient *data* (an encode-side arithmetic/padding bug) versus identical
//! data coded differently (an entropy-coding bug).
//!
//! Usage: `probe_i324_coeffs <a.jpg> <b.jpg>`

use libjpeg_turbo_rs::read_coefficients;

fn main() {
    let mut args = std::env::args().skip(1);
    let path_a: String = args.next().expect("usage: <a.jpg> <b.jpg>");
    let path_b: String = args.next().expect("usage: <a.jpg> <b.jpg>");

    let a = read_coefficients(&std::fs::read(&path_a).expect("read a")).expect("parse a");
    let b = read_coefficients(&std::fs::read(&path_b).expect("read b")).expect("parse b");

    println!("{path_a}: {}x{}", a.width, a.height);
    println!("{path_b}: {}x{}", b.width, b.height);

    for (component_index, (ca, cb)) in a.components.iter().zip(b.components.iter()).enumerate() {
        println!(
            "\ncomponent {component_index} (id {} / {}):",
            ca.component_id, cb.component_id
        );
        println!(
            "  blocks_x  {:>4} / {:<4}   blocks_y {:>4} / {:<4}   blocks {:>5} / {}",
            ca.blocks_x,
            cb.blocks_x,
            ca.blocks_y,
            cb.blocks_y,
            ca.blocks.len(),
            cb.blocks.len()
        );
        if ca.blocks_x != cb.blocks_x || ca.blocks_y != cb.blocks_y {
            println!("  ^ BLOCK GRID DIFFERS");
            continue;
        }
        let mut differing_blocks: Vec<usize> = Vec::new();
        for (index, (ba, bb)) in ca.blocks.iter().zip(cb.blocks.iter()).enumerate() {
            if ba != bb {
                differing_blocks.push(index);
            }
        }
        if differing_blocks.is_empty() {
            println!("  coefficients IDENTICAL -> divergence is entropy coding only");
        } else {
            println!(
                "  {} of {} blocks differ -> coefficient data differs",
                differing_blocks.len(),
                ca.blocks.len()
            );
            for &index in differing_blocks.iter().take(4) {
                let (row, col) = (index / ca.blocks_x, index % ca.blocks_x);
                println!("    block {index} (row {row}, col {col}):");
                let first: Vec<String> = (0..64)
                    .filter(|&k| ca.blocks[index][k] != cb.blocks[index][k])
                    .take(8)
                    .map(|k| format!("[{k}] {} vs {}", ca.blocks[index][k], cb.blocks[index][k]))
                    .collect();
                println!("      {}", first.join(", "));
            }
        }
    }
}
