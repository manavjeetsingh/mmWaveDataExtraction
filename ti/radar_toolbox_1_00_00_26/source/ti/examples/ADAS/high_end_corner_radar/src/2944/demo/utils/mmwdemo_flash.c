/**
 *   @file  mmwdemo_flash.c
 *
 *   @brief
 *      The file implements the functions which are required to access QSPI flash 
 *   from mmw demo.
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2021 Texas Instruments, Inc.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *    Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 *    Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the
 *    distribution.
 *
 *    Neither the name of Texas Instruments Incorporated nor the names of
 *    its contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 *  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 *  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 *  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 *  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 *  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 *  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/
/* Standard Include Files. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>

/* mmWave PDK Include Files: */
// #include <ti/drv/spi/SPI.h>
// #include <ti/drv/spi/soc/SPI_soc.h>
// #include <ti/board/src/flash/include/board_flash.h>

/* mmWave SDK Include Files: */
#include <ti/common/sys_common.h>
#include "mmwdemo_flash.h"

/* QSPI instance connected to OSPI NOR flash */
#define MMWDEMO_QSPI_NOR_INSTANCE                 (0U)
#define MMWDEMO_QSPI_FLASH_ID                     (BOARD_FLASH_ID_GD25B64CW2G)

/**************************************************************************
 **************************** Local Functions *****************************
 **************************************************************************/
typedef struct mmwDemo_Flash_t
{

    /*! @brief   QSPI flash driver handle */
    // Board_flashHandle QSPIFlashHandle;

    /*! @brief   Module initialized flag */
    bool              initialized;
}mmwDemo_Flash;


mmwDemo_Flash gMmwDemoFlash;

/**************************************************************************
 **************************** Monitor Functions *****************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      The function is used to initialize QSPI and Flash interface.
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t mmwDemo_flashInit(void)
{
#if 0
    QSPI_HwAttrs     QSPICfg;
    int32_t          retVal = 0;

    memset((void *)&gMmwDemoFlash, 0, sizeof(mmwDemo_Flash));

    retVal = QSPI_socGetInitCfg(MMWDEMO_QSPI_NOR_INSTANCE, &QSPICfg);

    QSPICfg.intrEnable = false;

    /* Set the default SPI init configurations */
    QSPI_socSetInitCfg(MMWDEMO_QSPI_NOR_INSTANCE, &QSPICfg);

    gMmwDemoFlash.QSPIFlashHandle = Board_flashOpen(MMWDEMO_QSPI_FLASH_ID, 
                                                MMWDEMO_QSPI_NOR_INSTANCE, NULL);

    if(gMmwDemoFlash.QSPIFlashHandle == NULL)
    {
        retVal = MMWDEMO_FLASH_EINVAL__QSPI;
        goto exit;
    }

    gMmwDemoFlash.initialized = true;

exit:
#endif
    return 0;
}

/**
 *  @b Description
 *  @n
 *      The function is used to close Flash interface.
 *
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
void mmwDemo_flashClose(void)
{
#if 0
    gMmwDemoFlash.initialized = false;

    /* Graceful shutdown */
    Board_flashClose(gMmwDemoFlash.QSPIFlashHandle);
#endif
    return;
}

/**
 *  @b Description
 *  @n
 *      The function is used to read data from flash.
 *
 *  @param[in]  flashOffset
 *      Flash Offset to read data from 
 *  @param[in]  readBuf
 *      Pointer to buffer that hold data read from flash
 *  @param[in]  size
 *      Size in bytes to be read from flash 
 *
 *  @pre
 *      mmwDemo_flashInit
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t mmwDemo_flashRead(uint32_t flashOffset, uint8_t *readBuf, uint32_t size)
{
#if 0
    int32_t retVal = 0;
    uint32_t          ioMode = BOARD_FLASH_QSPI_IO_MODE_QUAD; /* QSPI flash read/write access on four I/O lines */

    if(gMmwDemoFlash.initialized == true)
    {
        /* Read flash memory */
        if (Board_flashRead(gMmwDemoFlash.QSPIFlashHandle, flashOffset, readBuf,
                            size, (void *)(&ioMode)))
        {
            retVal = MMWDEMO_FLASH_EINVAL__QSPIFLASH;
        }
    }
    else
    {
        retVal = MMWDEMO_FLASH_EINVAL;
    }
#endif
    return 0;
}

/**
 *  @b Description
 *  @n
 *      The function is used to write data to flash.
 *
 *  @param[in]  flashOffset
 *      Flash Offset to write data to 
 *  @param[in]  writeBuf
 *      Pointer to buffer that hold data to be written to flash
 *  @param[in]  size
 *      Size in bytes to be written to flash 
 *
 *  @pre
 *      mmwDemo_flashInit
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t mmwDemo_flashWrite(uint32_t flashOffset, uint8_t *writeBuf, uint32_t size)
{
#if 0
    int32_t           retVal = 0;
    uint32_t          blockNum = 0;     /* flash block number */
    uint32_t          pageNum = 0;      /* flash page number */
    uint32_t          ioMode = BOARD_FLASH_QSPI_IO_MODE_QUAD; /* QSPI flash read/write access on four I/O lines */

    if(gMmwDemoFlash.initialized == true)
    {
        if(mmwDemo_flashEraseOneSector(flashOffset, &blockNum, &pageNum) < 0)
        {
            retVal = MMWDEMO_FLASH_EINVAL__QSPIFLASH;
        }
        else
        {
            /* Write buffer to flash */
            if (Board_flashWrite(gMmwDemoFlash.QSPIFlashHandle, flashOffset, writeBuf,
                                size, (void *)(&ioMode)))
            {
                retVal = MMWDEMO_FLASH_EINVAL__QSPIFLASH;
            }
        }
    }
    else
    {
        retVal = MMWDEMO_FLASH_EINVAL;
    }
#endif
    return 0;
}

/**
 *  @b Description
 *  @n
 *      The function is used to write data to flash.
 *
 *  @param[in]  flashOffset
 *      Flash Offset to write data to.
 *  @param[out]  blockNum
 *      Flash block number returned based on flash offset.
 *  @param[out]  pageNum
 *      Flash page number returned based on flash offset.
 *
 *  @pre
 *      mmwDemo_flashInit
 *
 *  @retval
 *      Success -   0
 *  @retval
 *      Error   -   <0
 */
int32_t mmwDemo_flashEraseOneSector(uint32_t flashOffset, uint32_t* blockNum, uint32_t* pageNum)
{
#if 0
    int32_t           retVal = 0;

    if(gMmwDemoFlash.initialized == true)
    {
        if (Board_flashOffsetToBlkPage(gMmwDemoFlash.QSPIFlashHandle, flashOffset,
                                       blockNum, pageNum))
        {
            retVal = MMWDEMO_FLASH_EINVAL__QSPIFLASH;
        }
        else
        {
            /* Erase block, to which data has to be written */
            if (Board_flashEraseBlk(gMmwDemoFlash.QSPIFlashHandle, *blockNum))
            {
                retVal = MMWDEMO_FLASH_EINVAL__QSPIFLASH;
            }
        }
    }
    else
    {
        retVal = MMWDEMO_FLASH_EINVAL;
    }
#endif
    return 0;
}

