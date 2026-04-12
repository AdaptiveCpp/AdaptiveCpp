# AMD ROCm Installation Guide

AdaptiveCpp supports AMD GPUs via the ROCm stack. Support is provided through the **generic single-pass compiler** (recommended) or the **HIP interoperability compiler**.

## Requirements

- **ROCm**: Version 4.0 or later (ROCm >= 5.3 recommended).
- **HIP**: Runtime libraries and headers must be installed.
- **Hardware**: Any AMD GPU supported by your chosen ROCm version.

---

## Path 1: Generic Single-Pass Compiler (Recommended)

This is the primary way to use AdaptiveCpp. It requires a compatible LLVM version.

> [!IMPORTANT]
> The LLVM version used to build AdaptiveCpp must be **less than or equal to** the LLVM version that ships with your ROCm installation.

- **Recommended ROCm version**: 5.3 or later.
- **Targets**: Use `--acpp-targets=generic` with your application.

---

## Path 2: HIP Interoperability Compiler

This path uses the legacy interoperability compiler. It may require specific Clang versions depending on the ROCm version (e.g., ROCm 4.5 requires Clang 13+).

- **Targets**: Use `--acpp-targets=hip` with your application.

---

## Using LLVM from ROCm (Not Recommended)

While you can build AdaptiveCpp against the LLVM bundled with ROCm, it is **not recommended** because ROCm's LLVM is often a vendor fork that lacks features present in official LLVM releases. This can lead to:
- Reduced kernel performance.
- Loss of SSCP/generic compiler support.

---

## CMake Configuration

Pass these variables during the AdaptiveCpp build to configure ROCm support:

| Variable | Description |
| :--- | :--- |
| `-DWITH_ROCM_BACKEND=ON` | Force enable the ROCm backend. |
| `-DROCM_PATH` | Path to the ROCm installation (Default: `/opt/rocm`). |

### Verification
After installation, verify that AdaptiveCpp can see your AMD GPU:
```bash
acpp-info
```

