# Sprout project rules

Rust + Vulkano GPU 2D multi-fluid sandbox. Build with `export PATH=$HOME/.cargo/bin:$PATH`;
`cargo build --release` before `./target/release/sprout validate` (tests don't rebuild the binary).

## Physics and stabilisation

- Fix causes first: unconverged solves, energy pumps and leaks get fixed at the source.
- Practical stabilisation is allowed when it buys speed or robustness (particle speed cap,
  light numerical damping, sleeping settled particles, clamps, higher CFL). Each must be a
  named parameter in `src/sim/params.rs` that can be switched off, and must be listed in the
  README's stabilisation table with its measured effect on speed and validation numbers.
- Not allowed: factors that exist only to make a test number match, or damping strong enough
  to visibly change how a fluid behaves.
- Material parameters stay centralised, documented and editable.

## Measuring

- Benchmark old and new builds with identical commands (`--no-vsync --frames N` with
  `SPROUT_FRAMES_CSV`); report frame-time percentiles and simulated time per wall second.
- Check `nvidia-smi` first; if a game holds the GPU, ask the user to close it.
- Done means measured improvement, with `validate`, `--selftest`, `cargo test --release` and
  `cargo clippy --release` passing. Commit only when the user asks.
