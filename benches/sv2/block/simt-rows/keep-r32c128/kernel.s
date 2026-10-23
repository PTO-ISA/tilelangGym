	.text
	.file	"kernel.asc"
	.globl	main_kernel                     // -- Begin function main_kernel
	.p2align	2
	.type	main_kernel,@function
main_kernel:                            // @main_kernel
	.cfi_startproc
// %bb.0:
	sub	sp, sp, #32
	.cfi_def_cfa_offset 32
	stp	x29, x30, [sp, #16]             // 16-byte Folded Spill
	add	x29, sp, #16
	.cfi_def_cfa w29, 16
	.cfi_offset w30, -8
	.cfi_offset w29, -16
	mov	x8, x2
	mov	w5, w1
	mov	w1, w0
	stp	x3, x4, [sp]
	adrp	x0, main_kernel
	add	x0, x0, :lo12:main_kernel
	adrp	x6, .L__unnamed_1
	add	x6, x6, :lo12:.L__unnamed_1
	mov	x2, sp
	mov	w3, #16
	mov	x4, x8
	bl	__cce_rtKernelEagerLaunch
	.cfi_def_cfa wsp, 32
	ldp	x29, x30, [sp, #16]             // 16-byte Folded Reload
	add	sp, sp, #32
	.cfi_def_cfa_offset 0
	.cfi_restore w30
	.cfi_restore w29
	ret
.Lfunc_end0:
	.size	main_kernel, .Lfunc_end0-main_kernel
	.cfi_endproc
                                        // -- End function
	.type	.L__unnamed_1,@object           // @0
	.section	.rodata.str1.1,"aMS",@progbits,1
.L__unnamed_1:
	.asciz	"main_kernel"
	.size	.L__unnamed_1, 12

	.type	__gAscCompile,@object           // @__gAscCompile
	.bss
	.weak	__gAscCompile
	.p2align	2
__gAscCompile:
	.byte	0                               // 0x0
	.size	__gAscCompile, 1

	.section	".linker-options","e",@llvm_linker_options
	.ident	"2026-08-18T19:12:48+08:00 clang version 15.0.5 (clang-5c68a1cb1231 flang-5c68a1cb1231)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym main_kernel
