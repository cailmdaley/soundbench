//! Mono WAV output, as `tools/lib/wav.ts` wrote it: 16-bit PCM, non-finite samples as 0, clamped
//! to [−1, 1], `Math.round(v · 32767)`. `--float` writes 32-bit IEEE float instead (the engine's
//! samples exactly, for comparisons).

use chimera::math::js_round;
use std::path::Path;

pub fn write(path: &str, x: &[f32], sample_rate: u32, float: bool) -> std::io::Result<()> {
    let (format, bytes_per_sample): (u16, u32) = if float { (3, 4) } else { (1, 2) };
    let data_bytes = x.len() as u32 * bytes_per_sample;
    let mut b = Vec::with_capacity(44 + data_bytes as usize);
    b.extend_from_slice(b"RIFF");
    b.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    b.extend_from_slice(b"WAVEfmt ");
    b.extend_from_slice(&16u32.to_le_bytes());
    b.extend_from_slice(&format.to_le_bytes());
    b.extend_from_slice(&1u16.to_le_bytes());
    b.extend_from_slice(&sample_rate.to_le_bytes());
    b.extend_from_slice(&(sample_rate * bytes_per_sample).to_le_bytes());
    b.extend_from_slice(&(bytes_per_sample as u16).to_le_bytes());
    b.extend_from_slice(&(bytes_per_sample as u16 * 8).to_le_bytes());
    b.extend_from_slice(b"data");
    b.extend_from_slice(&data_bytes.to_le_bytes());
    for &s in x {
        if float {
            b.extend_from_slice(&s.to_le_bytes());
        } else {
            let mut v = s as f64;
            if !v.is_finite() {
                v = 0.0;
            }
            v = v.clamp(-1.0, 1.0);
            b.extend_from_slice(&(js_round(v * 32767.0) as i16).to_le_bytes());
        }
    }
    if let Some(dir) = Path::new(path).parent()
        && !dir.as_os_str().is_empty()
    {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(path, b)
}
