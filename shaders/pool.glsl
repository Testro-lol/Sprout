// spray.comp's pool of bubbles, foam, spray and specks: handing out and returning entries
// (README, "Bubbles and foam"). The includer declares `dctl` (binding 95):
//   [0], [1]  entries of each part never handed out yet (bubbles, specks): a high-water mark
//   [2], [3]  entries of each part returned and free again: the height of its free stack
//   [8 ...]   the sprites' draw list (POOL entries), then the free stacks (bubbles' part at
//             its base, specks' at theirs).
// An emitter takes from the free stack first, then from the never-used entries; only a pool
// that is really full drops what it cannot hold. (Until 2026-11-17 an emitter probed one
// entry of a ring and dropped its bubble when that entry was still alive: once long-lived
// foam and trapped bubbles held much of the ring, new pours made few or none for seconds.)
// Taking (makers) and returning (move) run in different dispatches, so a stack is only ever
// popped or pushed at once.

// An entry of part `part` (0 bubbles, 1 specks; `base` its first entry, `cap` its size,
// `pool` the whole pool's), or -1.
int pool_take(uint part, uint base, uint cap, uint pool) {
    int top = int(atomicAdd(dctl[2u + part], 0xffffffffu));
    if (top > 0) return int(dctl[8u + pool + base + uint(top - 1)]);
    atomicAdd(dctl[2u + part], 1u);
    uint hw = atomicAdd(dctl[part], 1u);
    return hw < cap ? int(base + hw) : -1;
}

// Returns entry `i` of part `part` to its free stack.
void pool_return(uint part, uint base, uint pool, uint i) {
    uint k = atomicAdd(dctl[2u + part], 1u);
    dctl[8u + pool + base + k] = i;
}
