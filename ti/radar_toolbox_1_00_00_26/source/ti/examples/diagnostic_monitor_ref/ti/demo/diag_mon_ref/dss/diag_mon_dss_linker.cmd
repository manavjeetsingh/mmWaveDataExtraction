/******************************************************************************
 * C674 Bare Metal Operating System Linker Command File.
 *
 * Operating system are responsible for the following:-
 *  1. Installation of the reset vectors.
 *  2. Default placement of the various sections
 *  3. C674 Cache configuration
 *
 * This is a reference implementation which showcases usage of the platform
 * linker command file
 ******************************************************************************/

/******************************************************************************
 * Linker Options:
 ******************************************************************************/
--retain="*(.intvecs)"

--stack=8192
--heap=4096

/* L1P & L1D Cache sizes are set to be 16K. */
#define L1P_CACHE_SIZE      (16*1024)
#define L1D_CACHE_SIZE      (16*1024)

/******************************************************************************
 * Memory Map:
 *  L1 Memory accounts for the cache which is typically controlled by the
 *  operating system.
 ******************************************************************************/
MEMORY
{
#if (L1P_CACHE_SIZE < 0x8000)
    L1PSRAM:        o = 0x00E00000, l = (0x00008000 - L1P_CACHE_SIZE)
#endif
#if (L1D_CACHE_SIZE < 0x8000)
    L1DSRAM:        o = 0x00F00000, l = (0x00008000 - L1D_CACHE_SIZE)
#endif
}

/******************************************************************************
 * Section Configuration:
 ******************************************************************************/
SECTIONS
{
    /* Interrupt vector  */
    .intvecs:
    {
        . = align(32);
    } > L2SRAM_UMAP1

    .fardata:  {} >> L2SRAM_UMAP0 | L2SRAM_UMAP1
    .const:    {} >> L2SRAM_UMAP0 | L2SRAM_UMAP1
    .switch:   {} >> L2SRAM_UMAP0 | L2SRAM_UMAP1
    .cio:      {} >> L2SRAM_UMAP0 | L2SRAM_UMAP1
    .data:     {} >> L2SRAM_UMAP0 | L2SRAM_UMAP1
    .rodata:   {} >  L2SRAM_UMAP0 | L2SRAM_UMAP1
    .bss:      {} >  L2SRAM_UMAP0 | L2SRAM_UMAP1
    .neardata: {} >  L2SRAM_UMAP0 | L2SRAM_UMAP1
    .stack:    {} >  L2SRAM_UMAP0 | L2SRAM_UMAP1
    .cinit:    {} >  L2SRAM_UMAP0 | L2SRAM_UMAP1
    .far:      {} >  L2SRAM_UMAP0 | L2SRAM_UMAP1
    .text:     {} >> L2SRAM_UMAP1 | L2SRAM_UMAP0
    .sysmem :  {} >  L2SRAM_UMAP0

    /* for SDL related Diagnostic test */
    diag_data                   : {} > L2SRAM_UMAP0
    dataTxfrRAMECC_diag_data:   : {} > DATATXFRRAM
    HSRAMECC_diag_data:         : {} > HSRAM
    l2ecc_diag_code             : {} >> L2SRAM_UMAP1
    l1p_diag_data   			: {} > L1PSRAM
    l2_diag_data    			: {} >> L2SRAM_UMAP0 | L2SRAM_UMAP1
    l3ecc_diag_data:            : {} >> L3SRAM
    l2parity_diag_data_umap0:   : {} >> L2SRAM_UMAP0
    l2parity_diag_data_umap1:   : {} >> L2SRAM_UMAP1

}
/*----------------------------------------------------------------------------*/
