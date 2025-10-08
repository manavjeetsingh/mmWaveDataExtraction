/*
 * OSAL C674 ESM Driver Interface
 *
 * This file implements the C674 OSAL ESM Driver Interface
 *
 * Copyright (C) 2019 Texas Instruments Incorporated - http://www.ti.com/
 * ALL RIGHTS RESERVED
 *
 */

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/

/* Standard Include Files. */
#include <string.h>

/* mmWave SDL Include Files */
#include "osal/osal.h"
#include <ti/drivers/esm/esm.h>
#include <ti/drivers/esm/include/esm_internal.h>

/**************************************************************************
 **************************** Local Declarations **************************
 **************************************************************************/

void interrupt ESM_highpriority_FIQ(void);
void ESM_lowpriority_IRQ(uintptr_t arg);

/**************************************************************************
 **************************** Extern Declarations *************************
 **************************************************************************/

/* This function is used only by the Test code to initialize the OSAL R4F
 * ESM Driver module. This is the reason why it has not been added to the
 * osal.h. The osal.h will only have a list of all the exported functions
 * used by the diagnostics and which need to be ported by the customer. */
extern void OSAL_C674_ESMDrv_init (void);

/**************************************************************************
 **************************** Local Definitions ***************************
 **************************************************************************/

/**
 * @brief   Maximum number of Group1 ESM Error channels: As per the TRM
 * this is limited to 64
 */
#define ESM_MAX_GROUP1_ERROR_CHANNEL    (64U)

/**
 * @brief   Maximum number of Group2 ESM Error channels: As per the TRM
 * this is limited to 32
 */
#define ESM_MAX_GROUP2_ERROR_CHANNEL    (32U)

/**************************************************************************
 ***************************** Local Structures ***************************
 **************************************************************************/
#define MAX_ESM_NOTIFIER     4

typedef struct esmNotifierMapper
{
    uint32_t  errorNumber;
    uint32_t  notifierIdx;
}esmNotifierMapper_t;

esmNotifierMapper_t esmNotifMap[MAX_ESM_NOTIFIER] = {0};

ESM_Handle  gEsmHandle;

/**************************************************************************
 ******************************** Functions *******************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      In a working system the ESM handlers are owned by the
 *      application. However when a diagnostic is executed it
 *      would require a callback ("Hook") to be invoked by the
 *      application ESM handler. This is needed since the diagnostic
 *      will perform the necessary book keeping in the context
 *      of this function. Hooks are added during the start of
 *      the diagnostic execution.
 *
 *      The function is used to install a hook function for
 *      the ESM Group/Error number.
 *
 *  @param[in] ptrErrorChannel
 *      Pointer to the ESM error channel for which the notify function
 *      is to be installed.
 *  @param[in] fxn
 *      Error Hook Function which is invoked once the error is detected
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_ESMDrv_addHook (const CSL_ESM_ErrorChannel* ptrErrorChannel, OSAL_HookFxn fxn)
{
    /* For TI-RTOS, it uses mmWave-SDK ESM driver's register Notifier function */
    ESM_NotifyParams        notifyParams;
    int32_t                 errCode = 0;
    int32_t                 retVal = 0;
    uint8_t  i;

    retVal = 0;
    notifyParams.groupNumber = (uint32_t)ptrErrorChannel->group;
    notifyParams.errorNumber = (uint32_t)ptrErrorChannel->error;
    notifyParams.arg = NULL;
    notifyParams.notify = (ESM_CallBack)fxn;
    retVal = ESM_registerNotifier (gEsmHandle, &notifyParams, &errCode);
    
    for(i = 0; i < MAX_ESM_NOTIFIER; i++)
    {
        if(esmNotifMap[i].errorNumber == 0)
        {
            esmNotifMap[i].errorNumber = ptrErrorChannel->error;
            esmNotifMap[i].notifierIdx = retVal;
            break;
        }
    }
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the previous installed hook for
 *      the ESM Group/Error number
 *
 *  @param[in] ptrErrorChannel
 *      Pointer to the ESM error channel information for which
 *      the notify function is needed.
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_ESMDrv_delHook (const CSL_ESM_ErrorChannel* ptrErrorChannel)
{
    /* For TI-RTOS, it uses mmWave-SDK ESM driver's De-register Notifier function */
    int32_t     errCode = 0;
    uint8_t     i;

    /* find the notifier index registered for ESM error num */
    for(i = 0; i < MAX_ESM_NOTIFIER; i++)
    {
        if(esmNotifMap[i].errorNumber == (uint32_t)ptrErrorChannel->error)
            break;        
    }
    
    ESM_deregisterNotifier (gEsmHandle, esmNotifMap[i].notifierIdx, &errCode);

    esmNotifMap[i].errorNumber = 0;
    esmNotifMap[i].notifierIdx = 0;
}

/**
 *  @b Description
 *  @n
 *      The function is used to initialize the ESM Driver Interface.
 *
 *      This is a reference only implementation which is invoked
 *      only from the Unit Tests.
 *
 *  \ingroup OSAL_ESM_DRV_INTERFACE
 *
 *  @retval
 *      Not applicable
 *
 *  @note
 *      The function is used to initialize the ESM Driver Interface.
 *      The ESM Driver interface should only be invoked once the
 *      Interrupt module has been initialized since this will
 *      register error ISR
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_ESMDrv_init (void)
{
    /* For TI-RTOS, it uses mmWave-SDK ESM driver's Init function */
    uint8_t bClearErrors = 0;
    HwiP_Params     hwiParams;
    extern ESM_DriverMCB   gEsmMCB;

    /* use mmwave SDK ESM driver initialization here */
    gEsmHandle = ESM_init(bClearErrors);  

    /* this is hooked with NMI interrupt handler in *.cfg */
    //ESM_highpriority_FIQ

    HwiP_Params_init(&hwiParams);
    hwiParams.name = "ESM_IRQ";
    hwiParams.type = HwiP_Type_IRQ;
    gEsmMCB.hwiHandleLo = HwiP_create(32 /*CSL_XWR16XX_C674_ESM_LOW_PRIORITY_EVENT*/, (HwiP_Fxn)ESM_lowpriority_IRQ, &hwiParams);


    return;
}

