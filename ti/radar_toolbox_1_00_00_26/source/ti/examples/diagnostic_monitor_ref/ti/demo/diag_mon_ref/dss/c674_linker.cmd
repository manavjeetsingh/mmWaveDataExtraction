/******************************************************************************
 * File: c674_linker.cmd
 *
 * Description:
 * C674 XWR18xx Linker command file which specifies the memory map
 ******************************************************************************/

/******************************************************************************
 * Memory Map:
 *  The L1 Memory can be divided depending upon the cache sizes. Caches are
 *  controlled by the operating system. This implies that we need to define the
 *  L1 memory map in the OSAL Linker command file.
 ******************************************************************************/
MEMORY
{
    L2SRAM_UMAP1:   o = 0x007E0000, l = 0x00020000
    L2SRAM_UMAP0:   o = 0x00800000, l = 0x00020000
    L3SRAM:         o = 0x20000000, l = MMWAVE_L3RAM_SIZE
    DATATXFRRAM:    o = 0x21078000, l = 0x2000
    HSRAM:          o = 0x21080000, l = 0x8000
}
/*----------------------------------------------------------------------------*/
