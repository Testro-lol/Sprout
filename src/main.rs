//! Sprout: a fast, interactive 2D multi-fluid sandbox.

mod app;
mod gpu;
mod gui;
mod headless;
mod maps;
mod pacing;
mod render;
mod scene;
mod selftest;
mod sim;
mod ui;
mod validate;
mod world;

use gpu::Gpu;

const USAGE: &str = "\
Sprout - 2D multi-fluid sandbox

USAGE:
    sprout [OPTIONS]                 run the interactive sandbox
    sprout shot OUT.png [OPTIONS]    render a map off-screen
    sprout bench [OPTIONS]           measure simulation and rendering throughput
    sprout validate [TEST]           run the physics and tool checks (hydrostatic, dam, ritter, layers,
                                     viscous, buoyancy, tools)
    sprout --selftest                drive the real window through every tool and control
    sprout maps                      list the built-in maps

OPTIONS:
    --map <N|name>        starting map (default 0)
    --quality <q>         grid resolution: low (384x216), medium (512x288, default),
                          high (768x432), ultra (1024x576)
    --grid <WxH>          explicit grid size in cells
    --size <WxH>          window / image size in pixels
    --at <t1,t2,...>      shot: simulated seconds at which to save frames (default 2)
                          bench: simulated seconds each scene runs before it is timed (default 6)
    --ui                  shot: include the control panel
    --shots <DIR>         validate: save a picture of each scenario
    --no-vsync            present as fast as possible (uncapped frame rate)
    --frames <N>          run N frames, print frame timings and exit
    --set <name=value>    override a physics parameter (repeatable), e.g. --set nu_syrup=900
";

fn parse_pair(s: &str) -> Option<[u32; 2]> {
    let (a, b) = s.split_once(['x', 'X'])?;
    Some([a.trim().parse().ok()?, b.trim().parse().ok()?])
}

pub struct Options {
    pub map: usize,
    pub grid: [u32; 2],
    pub size: Option<[u32; 2]>,
    pub times: Vec<f32>,
    pub positional: Vec<String>,
    pub vsync: bool,
    pub frames: Option<u64>,
    pub with_ui: bool,
    pub selftest: bool,
    /// validate: directory for a picture of each scenario.
    pub shots: Option<String>,
    pub grid_given: bool,
    pub times_given: bool,
    pub map_given: bool,
    /// `--set name=value` physics overrides.
    pub overrides: Vec<(String, f32)>,
}

impl Options {
    pub fn apply(&self, params: &mut sim::SimParams) {
        for (k, v) in &self.overrides {
            params.set(k, *v);
        }
    }
}

fn parse_options(args: &[String]) -> Result<Options, String> {
    let mut o = Options { map: 0, grid: [512, 288], size: None, times: vec![2.0], positional: Vec::new(), vsync: true, frames: None, with_ui: false, selftest: false, shots: None, grid_given: false, times_given: false, map_given: false, overrides: Vec::new() };
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().cloned().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--map" => {
                o.map_given = true;
                let v = value("--map")?;
                o.map = v
                    .parse::<usize>()
                    .ok()
                    .or_else(|| {
                        let v = v.to_lowercase().replace([' ', '-', '_'], "");
                        maps::MAPS.iter().position(|m| m.name.to_lowercase().replace(' ', "").starts_with(&v))
                    })
                    .filter(|&i| i < maps::MAPS.len())
                    .ok_or(format!("unknown map '{v}' (try `sprout maps`)"))?;
            }
            "--quality" => {
                o.grid_given = true;
                o.grid = match value("--quality")?.as_str() {
                    "low" => [384, 216],
                    "medium" => [512, 288],
                    "high" => [768, 432],
                    "ultra" => [1024, 576],
                    q => return Err(format!("unknown quality '{q}'")),
                }
            }
            "--grid" => {
                o.grid_given = true;
                o.grid = parse_pair(&value("--grid")?).ok_or("--grid expects WxH")?
            }
            "--size" => o.size = Some(parse_pair(&value("--size")?).ok_or("--size expects WxH")?),
            "--at" => {
                o.times_given = true;
                o.times = value("--at")?
                    .split(',')
                    .map(|t| t.trim().parse::<f32>().map_err(|_| format!("bad time '{t}'")))
                    .collect::<Result<_, _>>()?
            }
            "--set" => {
                let v = value("--set")?;
                let (k, x) = v.split_once('=').ok_or("--set expects name=value")?;
                let x: f32 = x.parse().map_err(|_| format!("bad value in '{v}'"))?;
                if !sim::SimParams::default().set(k, x) {
                    return Err(format!("unknown parameter '{k}'"));
                }
                o.overrides.push((k.to_string(), x));
            }
            "--no-vsync" => o.vsync = false,
            "--ui" => o.with_ui = true,
            "--selftest" => o.selftest = true,
            "--shots" => o.shots = Some(value("--shots")?),
            "--frames" => o.frames = Some(value("--frames")?.parse().map_err(|_| "--frames expects a number")?),
            "-h" | "--help" => return Err(String::new()),
            s if s.starts_with("--") => return Err(format!("unknown option '{s}'")),
            _ => o.positional.push(a.clone()),
        }
    }
    if o.grid[0] < 64 || o.grid[1] < 64 || o.grid[0] > 4096 || o.grid[1] > 4096 {
        return Err("grid must be between 64 and 4096 cells per side".into());
    }
    Ok(o)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (command, rest) = match args.first().map(String::as_str) {
        Some(c @ ("shot" | "maps" | "probe" | "bench" | "profile" | "validate")) => (c, &args[1..]),
        _ => ("run", &args[..]),
    };
    let o = match parse_options(rest) {
        Ok(o) => o,
        Err(e) => {
            if !e.is_empty() {
                eprintln!("error: {e}\n");
            }
            eprint!("{USAGE}");
            std::process::exit(if e.is_empty() { 0 } else { 2 });
        }
    };

    match command {
        "maps" => {
            for (i, m) in maps::MAPS.iter().enumerate() {
                println!("{i}  {:<14} {}", m.name, m.blurb);
            }
        }
        "shot" => {
            let out = o.positional.first().cloned().unwrap_or_else(|| "sprout.png".into());
            let gpu = Gpu::new(None);
            println!("device: {}", gpu.device_name());
            headless::shot(
                gpu,
                headless::ShotOptions {
                    map: o.map,
                    grid: o.grid,
                    size: o.size.unwrap_or([1280, 720]),
                    times: o.times.clone(),
                    out,
                    with_ui: o.with_ui,
                },
                &|p| o.apply(p),
            );
        }
        "bench" => {
            let gpu = Gpu::new(None);
            let grids: Vec<[u32; 2]> =
                if o.grid_given { vec![o.grid] } else { vec![[384, 216], [512, 288], [768, 432], [1024, 576]] };
            let maps: Vec<usize> = if o.map_given { vec![o.map] } else { vec![1, 0, 2] };
            // `--at` sets how long each scene runs before it is measured.
            let warmup = if o.times_given { o.times[0] } else { 6.0 };
            headless::bench(gpu, &|p| o.apply(p), &grids, &maps, o.size.unwrap_or([1920, 1080]), warmup);
        }
        "validate" => {
            let ok = validate::run_all(Gpu::new(None), &|p| o.apply(p), o.positional.first().map(String::as_str), o.shots.as_deref());
            std::process::exit(if ok { 0 } else { 1 });
        }
        "profile" => headless::profile(Gpu::new(None), &|p| o.apply(p), o.grid, o.map, o.times[0]),
        "probe" => {
            let gpu = Gpu::new(None);
            let depth: f32 = o.positional.first().and_then(|s| s.parse().ok()).unwrap_or(80.0);
            headless::probe(gpu, o.grid, &|p| o.apply(p), o.map_given.then_some(o.map), depth, o.times[0], o.positional.get(1).map(String::as_str));
        }
        _ => {
            let event_loop = winit::event_loop::EventLoop::new().expect("no display available");
            let gpu = Gpu::new(Some(&event_loop));
            println!("device: {}", gpu.device_name());
            // The self-test script is written for the empty tank at the default resolution,
            // and paces itself in frames, so it wants a capped frame rate.
            let ok = app::run(
                event_loop,
                gpu,
                app::AppOptions {
                    map: if o.selftest { maps::MAPS.len() - 1 } else { o.map },
                    grid: if o.selftest { [512, 288] } else { o.grid },
                    size: o.size.unwrap_or([1600, 900]),
                    vsync: o.vsync || o.selftest,
                    overrides: o.overrides,
                    frames: o.frames,
                    selftest: o.selftest,
                },
            );
            std::process::exit(if ok { 0 } else { 1 });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, String> {
        parse_options(&args.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn options_parse() {
        let o = parse(&["--quality", "high", "--map", "dam", "--set", "nu_syrup=900", "--no-vsync", "out.png"]).unwrap();
        assert_eq!(o.grid, [768, 432]);
        assert_eq!(maps::MAPS[o.map].name, "Dam Break");
        assert_eq!(o.overrides, vec![("nu_syrup".to_string(), 900.0)]);
        assert!(!o.vsync);
        assert_eq!(o.positional, vec!["out.png".to_string()]);

        let o = parse(&["--grid", "640x360", "--at", "1, 2.5", "--map", "2"]).unwrap();
        assert_eq!((o.grid, o.map), ([640, 360], 2));
        assert_eq!(o.times, vec![1.0, 2.5]);
    }

    #[test]
    fn bad_options_are_rejected() {
        assert!(parse(&["--map", "nowhere"]).is_err());
        assert!(parse(&["--map", "99"]).is_err());
        assert!(parse(&["--quality", "extreme"]).is_err());
        assert!(parse(&["--grid", "8x8"]).is_err());
        assert!(parse(&["--set", "bogus=1"]).is_err());
        assert!(parse(&["--set", "nu_syrup"]).is_err());
        assert!(parse(&["--frobnicate"]).is_err());
        assert!(parse(&["--size"]).is_err());
    }
}
