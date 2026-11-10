//! Compiles the GLSL in `shaders/` to SPIR-V with `glslc` (or `glslangValidator`).
//!
//! Sprout deliberately avoids `vulkano-shaders`: that crate builds the whole shaderc C++
//! toolchain from source, which is slow and breaks with recent CMake. Calling the system
//! compiler keeps clean builds under a minute.

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version").output().is_ok()
}

fn main() {
    println!("cargo:rerun-if-changed=shaders");
    println!("cargo:rerun-if-env-changed=GLSLC");

    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dir = Path::new("shaders");

    let glslc = env::var("GLSLC").ok().or_else(|| have("glslc").then(|| "glslc".to_string()));
    let use_glslang = glslc.is_none();
    if use_glslang && !have("glslangValidator") {
        panic!(
            "\n\nSprout needs a GLSL compiler at build time.\n\
             Install `glslc` (Fedora: `sudo dnf install glslc`, Debian/Ubuntu: `sudo apt install glslc`,\n\
             Arch: `sudo pacman -S shaderc`, or the Vulkan SDK) or point the GLSLC env var at one.\n\n"
        );
    }

    let mut entries: Vec<_> = fs::read_dir(dir).unwrap().map(|e| e.unwrap().path()).collect();
    entries.sort();
    for path in entries {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if !matches!(ext, "comp" | "vert" | "frag") {
            continue;
        }
        let name = path.file_name().unwrap().to_str().unwrap();
        let dst = out.join(format!("{name}.spv"));
        let mut cmd;
        if use_glslang {
            cmd = Command::new("glslangValidator");
            cmd.arg("-V").arg("--target-env").arg("vulkan1.1");
            cmd.arg(format!("-I{}", dir.display())).arg("-o").arg(&dst).arg(&path);
        } else {
            cmd = Command::new(glslc.as_ref().unwrap());
            cmd.arg("--target-env=vulkan1.1").arg("-O");
            cmd.arg("-I").arg(dir).arg("-o").arg(&dst).arg(&path);
        }
        let output = cmd.output().expect("failed to launch the GLSL compiler");
        if !output.status.success() {
            panic!(
                "\n\nshader compilation failed for {}:\n{}\n{}\n",
                path.display(),
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
