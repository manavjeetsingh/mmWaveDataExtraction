/*----------------------------------------------------------------------------*/
/* Linker Settings                                                            */
--retain="*(.intvecs)"

/*----------------------------------------------------------------------------*/
/* Section Configuration                                                      */
SECTIONS
{
    systemHeap : {} > DATA_RAM
    /* for SDL Diagnostic Test Purpose */
    diag_data       : {} > DATA_RAM
    csl_data        : {} > DATA_RAM
    .tcma    : {} > PROG_RAM
    .tcmb    : {} > DATA_RAM
    .l3ram   : {} > L3_RAM
    .hsram 	 : {} > HWA_RAM
    .exceptionStack : {} > DATA_RAM
}
/*----------------------------------------------------------------------------*/

