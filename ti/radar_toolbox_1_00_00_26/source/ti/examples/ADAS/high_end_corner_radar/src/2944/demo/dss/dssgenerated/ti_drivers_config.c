/*
 *  Copyright (C) 2021 Texas Instruments Incorporated
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
/*
 * Auto generated file - DO NOT MODIFY
 */

#include "ti_drivers_config.h"

/*
 * EDMA
 */
/* EDMA atrributes */
static EDMA_Attrs gEdmaAttrs[CONFIG_EDMA_NUM_INSTANCES] =
{
    {

        .baseAddr           = CSL_DSS_TPCC_A_U_BASE,
        .compIntrNumber     = CSL_DSS_INTR_DSS_TPCC_A_INTAGG,
        .intrAggEnableAddr  = CSL_DSS_CTRL_U_BASE + CSL_DSS_CTRL_DSS_TPCC_A_INTAGG_MASK,
        .intrAggEnableMask  = 0x1FF & (~(2U << 0)),
        .intrAggStatusAddr  = CSL_DSS_CTRL_U_BASE + CSL_DSS_CTRL_DSS_TPCC_A_INTAGG_STATUS,
        .intrAggClearMask   = (2U << 0),
        .initPrms           =
        {
            .regionId     = 0,
            .queNum       = 0,
            .initParamSet = FALSE,
            .ownResource    =
            {
                .qdmaCh      = 0x3FU,
                .dmaCh[0]    = 0xFFFFFFFFU,
                .dmaCh[1]    = 0xFFFFFFFFU,
                .tcc[0]      = 0xFFFFFFFFU,
                .tcc[1]      = 0xFFFFFFFFU,
                .paramSet[0] = 0xFFFFFFFFU,
                .paramSet[1] = 0xFFFFFFFFU,
                .paramSet[2] = 0xFFFFFFFFU,
                .paramSet[3] = 0xFFFFFFFFU,
            },
            .reservedDmaCh[0]    = 0x01U,
            .reservedDmaCh[1]    = 0x00U,
        },
    },
};

/* EDMA objects - initialized by the driver */
static EDMA_Object gEdmaObjects[CONFIG_EDMA_NUM_INSTANCES];
/* EDMA driver configuration */
EDMA_Config gEdmaConfig[CONFIG_EDMA_NUM_INSTANCES] =
{
    {
        &gEdmaAttrs[CONFIG_EDMA0],
        &gEdmaObjects[CONFIG_EDMA0],
    },
};

uint32_t gEdmaConfigNum = CONFIG_EDMA_NUM_INSTANCES;

/*
 * HWA
 */
/* HWA atrributes */
HWA_Attrs gHwaAttrs[CONFIG_HWA_NUM_INSTANCES] =
{
    {
        .instanceNum                = 0U,
        .ctrlBaseAddr               = CSL_DSS_HWA_CFG_U_BASE,
        .paramBaseAddr              = CSL_DSS_HWA_PARAM_U_BASE,
        .ramBaseAddr                = CSL_DSS_HWA_WINDOW_RAM_U_BASE,
        .dssBaseAddr                = CSL_DSS_CTRL_U_BASE,
        .numHwaParamSets            = SOC_HWA_NUM_PARAM_SETS,
        .intNum1ParamSet            = CSL_DSS_INTR_DSS_HWA_PARAM_DONE_INTR1,
        .intNum2ParamSet            = CSL_DSS_INTR_DSS_HWA_PARAM_DONE_INTR2,
        .intNumDone                 = CSL_DSS_INTR_DSS_HWA_LOOP_INTR1,
        .intNumDoneALT              = CSL_DSS_INTR_DSS_HWA_LOOP_INTR2,
        .numDmaChannels             = SOC_HWA_NUM_DMA_CHANNEL,
        .accelMemBaseAddr           = CSL_DSS_HWA_DMA0_U_BASE,
        .accelMemSize               = SOC_HWA_MEM_SIZE,
        .isConcurrentAccessAllowed  = true,
    },
};
/* HWA RAM atrributes */
HWA_RAMAttrs gHwaRamCfg[HWA_NUM_RAMS] =
{
    {CSL_DSS_HWA_WINDOW_RAM_U_BASE, CSL_DSS_HWA_WINDOW_RAM_U_SIZE},
    {CSL_DSS_HWA_MULT_RAM_U_BASE, CSL_DSS_HWA_MULT_RAM_U_SIZE},
    {CSL_DSS_HWA_DEROT_RAM_U_BASE, CSL_DSS_HWA_DEROT_RAM_U_SIZE},
    {CSL_DSS_HWA_SHUFFLE_RAM_U_BASE,CSL_DSS_HWA_SHUFFLE_RAM_U_SIZE},
    {CSL_DSS_HWA_HIST_THRESH_RAM_U_BASE, CSL_DSS_HWA_HIST_THRESH_RAM_U_SIZE},
    {CSL_DSS_HWA_2DSTAT_ITER_VAL_RAM_U_BASE, CSL_DSS_HWA_2DSTAT_ITER_VAL_RAM_U_SIZE},
    {CSL_DSS_HWA_2DSTAT_ITER_IDX_RAM_U_BASE, CSL_DSS_HWA_2DSTAT_ITER_IDX_RAM_U_SIZE},
    {CSL_DSS_HWA_2DSTAT_SMPL_VAL_RAM_U_BASE, CSL_DSS_HWA_2DSTAT_SMPL_VAL_RAM_U_SIZE},
    {CSL_DSS_HWA_2DSTAT_SMPL_IDX_RAM_U_BASE, CSL_DSS_HWA_2DSTAT_SMPL_IDX_RAM_U_SIZE},
    {CSL_DSS_HWA_HIST_RAM_U_BASE, CSL_DSS_HWA_HIST_RAM_U_SIZE}
};

/* HWA objects - initialized by the driver */
HWA_Object gHwaObject[CONFIG_HWA_NUM_INSTANCES];
/* HWA objects - storage for HWA driver object handles */
HWA_Object *gHwaObjectPtr[CONFIG_HWA_NUM_INSTANCES] = { NULL };
/* HWA objects count */
uint32_t gHwaConfigNum = CONFIG_HWA_NUM_INSTANCES;

/*
 * IPC Notify
 */
#include <drivers/ipc_notify.h>
#include <drivers/ipc_notify/v1/ipc_notify_v1.h>

/* this function is called within IpcNotify_init, this function returns core specific IPC config */
void IpcNotify_getConfig(IpcNotify_InterruptConfig **interruptConfig, uint32_t *interruptConfigNum)
{
    /* extern globals that are specific to this core */
    extern IpcNotify_InterruptConfig gIpcNotifyInterruptConfig_c66ss0[];
    extern uint32_t gIpcNotifyInterruptConfigNum_c66ss0;

    *interruptConfig = &gIpcNotifyInterruptConfig_c66ss0[0];
    *interruptConfigNum = gIpcNotifyInterruptConfigNum_c66ss0;
}

/*
 * IPC RP Message
 */
#include <drivers/ipc_rpmsg.h>

/* Number of CPUs that are enabled for IPC RPMessage */
#define IPC_RPMESSAGE_NUM_CORES           (2U)
/* Number of VRINGs for the numner of CPUs that are enabled for IPC */
#define IPC_RPMESSAGE_NUM_VRINGS          (IPC_RPMESSAGE_NUM_CORES*(IPC_RPMESSAGE_NUM_CORES-1))
/* Number of a buffers in a VRING, i.e depth of VRING queue */
#define IPC_RPMESSAGE_NUM_VRING_BUF       (1U)
/* Max size of a buffer in a VRING */
#define IPC_RPMESSAGE_MAX_VRING_BUF_SIZE  (1152U)
/* Size of each VRING is
 *     number of buffers x ( size of each buffer + space for data structures of one buffer (32B) )
 */
#define IPC_RPMESSAGE_VRING_SIZE          RPMESSAGE_VRING_SIZE(IPC_RPMESSAGE_NUM_VRING_BUF, IPC_RPMESSAGE_MAX_VRING_BUF_SIZE)

/* VRING base address, all VRINGs are put one after other in the below region.
 *
 * IMPORTANT: Make sure of below,
 * - The section defined below should be placed at the exact same location in memory for all the CPUs
 * - The memory should be marked as non-cached for all the CPUs
 * - The section should be marked as NOLOAD in all the CPUs linker command file
 */
/* In this case gRPMessageVringMem size is 2432 bytes */
uint8_t gRPMessageVringMem[IPC_RPMESSAGE_NUM_VRINGS][IPC_RPMESSAGE_VRING_SIZE] __attribute__((aligned(128), section(".bss.ipc_vring_mem")));




void Pinmux_init();
void PowerClock_init(void);
void PowerClock_deinit(void);

/*
 * Common Functions
 */
void System_init(void)
{
    /* DPL init sets up address transalation unit, on some CPUs this is needed
     * to access SCICLIENT services, hence this needs to happen first
     */
    Dpl_init();
    PowerClock_init();
    /* Now we can do pinmux */
    Pinmux_init();
    /* finally we initialize all peripheral drivers */
    EDMA_init();
    HWA_init();
    /* IPC Notify */
    {
        IpcNotify_Params notifyParams;
        int32_t status;

        /* initialize parameters to default */
        IpcNotify_Params_init(&notifyParams);

        /* specify the core on which this API is called */
        notifyParams.selfCoreId = CSL_CORE_ID_C66SS0;

        /* list the cores that will do IPC Notify with this core
        * Make sure to NOT list 'self' core in the list below
        */
        notifyParams.numCores = 1;
        notifyParams.coreIdList[0] = CSL_CORE_ID_R5FSS0_0;

        /* initialize the IPC Notify module */
        status = IpcNotify_init(&notifyParams);
        DebugP_assert(status==SystemP_SUCCESS);

        { /* Mailbox driver MUST be initialized after IPC Notify init */
            Mailbox_Params mailboxInitParams;

            Mailbox_Params_init(&mailboxInitParams);
            status = Mailbox_init(&mailboxInitParams);
            DebugP_assert(status == SystemP_SUCCESS);
        }
    }
    /* IPC RPMessage */
    {
        RPMessage_Params rpmsgParams;
        int32_t status;

        /* initialize parameters to default */
        RPMessage_Params_init(&rpmsgParams);

        /* VRING mapping from source core to destination core, '-1' means NO VRING,
            r5fss0_0 => {"r5fss0_0":-1,"c66ss0":0}
            c66ss0 => {"r5fss0_0":1,"c66ss0":-1}
         */
        /* TX VRINGs */
        rpmsgParams.vringTxBaseAddr[CSL_CORE_ID_R5FSS0_0] = (uintptr_t)gRPMessageVringMem[1];
        /* RX VRINGs */
        rpmsgParams.vringRxBaseAddr[CSL_CORE_ID_R5FSS0_0] = (uintptr_t)gRPMessageVringMem[0];
        /* Other VRING properties */
        rpmsgParams.vringSize = IPC_RPMESSAGE_VRING_SIZE;
        rpmsgParams.vringNumBuf = IPC_RPMESSAGE_NUM_VRING_BUF;
        rpmsgParams.vringMsgSize = IPC_RPMESSAGE_MAX_VRING_BUF_SIZE;

        /* initialize the IPC RP Message module */
        status = RPMessage_init(&rpmsgParams);
        DebugP_assert(status==SystemP_SUCCESS);
    }

}

void System_deinit(void)
{
    EDMA_deinit();
    HWA_deinit();
    RPMessage_deInit();
    IpcNotify_deInit();

    PowerClock_deinit();
    Dpl_deinit();
}
