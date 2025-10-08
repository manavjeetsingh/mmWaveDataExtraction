/*
 * C674 OSAL Interrupt Management
 *
 * This file implements the C674 OSAL Interrupt Management.
 *
 * Copyright (C) 2019 Texas Instruments Incorporated - http://www.ti.com/
 * ALL RIGHTS RESERVED
 *
 */

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/

/* Standard Include Files. */
#include <stdint.h>
#include <c6x.h>

/* mmWave SDL Include Files */
#include "osal/osal.h"
#include <ti/drivers/osal/HwiP.h>
#include <ti/sysbios/knl/Task.h>
#include <ti/sysbios/family/c64p/Hwi.h>
#include <ti/sysbios/family/c64p/EventCombiner.h>


/**************************************************************************
 **************************** Local Declarations **************************
 **************************************************************************/

/* This function is used only by the Test code to initialize the OSAL C674
 * Interrupt module. This is the reason why it has not been added to the
 * osal.h. The osal.h will only have a list of all the exported functions
 * used by the diagnostics and which need to be ported by the customer. */
extern void OSAL_C674_Interrupt_init (void);

/**************************************************************************
 ***************************** Local Structures ***************************
 **************************************************************************/


#define MAX_HWI_HANDLE     4
typedef struct hwiHandleMap_t
{
    HwiP_Handle hwiHandle;
    uint8_t     irq;
}hwiHandleMap;


hwiHandleMap gHwiHandleMap[MAX_HWI_HANDLE] = {0};

/**
 * @brief
 *  R4F Interrupt Vector Table
 *
 * @details
 *  The structure is used to hold information of the registered
 *  interrupt handlers which are invoked on the reception of the
 *  R4F interrupt.
 *
 *  \ingroup OSAL_INTERNAL_STRUCTURES
 */
typedef struct OSAL_C674_Interrupt_VectorTable_t
{
    /**
     * @brief   Undefined Instruction Handler
     */
    OSAL_HookFxn    undefInstrHandler;

    /**
     * @brief   Data Abort Handler
     */
    OSAL_HookFxn    dataAbortHandler;

    /**
     * @brief   SVC Handler
     */
    OSAL_HookFxn    svcHandler;

    /**
     * @brief   Prefetch Abort Handler
     */
    OSAL_HookFxn    prefetchAbortHandler;
}OSAL_C674_Interrupt_VectorTable;

/**************************************************************************
 ******************************** Functions *******************************
 **************************************************************************/
static void OSAL_C674_Interrupt_defaultHandler (void);


/**
 * @brief   R4F Interrupt Vector Table:
 */
OSAL_C674_Interrupt_VectorTable  gOSAL_C674_InterruptVectorTable =
{
    OSAL_C674_Interrupt_defaultHandler,  /* Undefined Instruction Handler:   */
    OSAL_C674_Interrupt_defaultHandler,  /* Data Abort Handler:              */
    OSAL_C674_Interrupt_defaultHandler,  /* SVC Handler:                     */
#if defined (DYNAMIC_ANALYSIS_ENABLED)
    OSAL_C674_Interrupt_emptyHandler    /* Prefetch Abort Handler:          */
#else
    OSAL_C674_Interrupt_defaultHandler   /* Prefetch Abort Handler:          */
#endif
};

/**************************************************************************
 ******************************** Functions *******************************
 **************************************************************************/
#if 0
/* For TI-RTOS (SysBios), user-defined exception handler is attached with OS
 * so this function gets called instead of SysBios Internal exception handler.
 * This handler function is attached with OS by mmw_mss.cfg.
 *
 * Based on different Diagnostic test, it connects its own handler function
 * before generating any exception (like CPU data abort) during diagnostic test.
 */
void MMW_EXCEPTION_HANDLER(Exception_ExcContext *excContext)
{
    /* check type of exception and call relevant handler based on that */
    if(excContext->type == ti_sysbios_family_arm_exc_Exception_Type_PreAbort)
    {
        gOSAL_C674_InterruptVectorTable.prefetchAbortHandler();
    }
    else if(excContext->type == ti_sysbios_family_arm_exc_Exception_Type_DataAbort)
    {
        gOSAL_C674_InterruptVectorTable.dataAbortHandler();
    }
    else if(excContext->type == ti_sysbios_family_arm_exc_Exception_Type_UndefInst)
    {
        gOSAL_C674_InterruptVectorTable.undefInstrHandler ();
    }
    else if(excContext->type == ti_sysbios_family_arm_exc_Exception_Type_Supervisor)
    {
        /* ti_sysbios_family_arm_exc_Exception_Type_Supervisor */
        gOSAL_C674_InterruptVectorTable.svcHandler();
    }
    else
    {
        excContext->type = (ti_sysbios_family_arm_exc_Exception_Type)0;
    }

}
#endif
/**
 *  @b Description
 *  @n
 *      Default Interrupt/CPU exception handler which is installed
 *      for all the interrupts & CPU exceptions. The function will
 *      get stuck in a loop indefinitely. This was done to catch
 *      all unexpected interrupts/CPU exceptions
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 */
static void OSAL_C674_Interrupt_defaultHandler (void)
{
    volatile uint32_t deadloop = 1U;
    while (deadloop == 1U)
    {
    }
}

#if 0
/**
 *  @b Description
 *  @n
 *      In a working system the exceptions are owned by the application
 *      However when a diagnostic is executed it would require a
 *      callback ("Hook") to be invoked by the application exception
 *      handler. This is needed since the diagnostic will perform
 *      the necessary book keeping in the context of this function.
 *      Hooks are added during the start of the diagnostic execution.
 *
 *      The function is used to install a hook to the R4F CPU
 *      exception handler. The hook function will be triggered
 *      on the CPU exception.
 *
 *  @param[in] type
 *      Exception Type
 *  @param[in] fxn
 *      Exception Hook to be installed
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_C674_Interrupt_addExceptionHook (OSAL_C674_Exception type, OSAL_HookFxn fxn)
{
    switch (type)
    {
        case OSAL_C674_Exception_PREFETCH_ABORT:
        {
            gOSAL_C674_InterruptVectorTable.prefetchAbortHandler = fxn;
            break;
        }
        case OSAL_C674_Exception_DATA_ABORT:
        {
            gOSAL_C674_InterruptVectorTable.dataAbortHandler = fxn;
            break;
        }
        case OSAL_C674_Exception_UNDEFINED_INSTR:
        {
            gOSAL_C674_InterruptVectorTable.undefInstrHandler = fxn;
            break;
        }
        case OSAL_C674_Exception_SVC:
        {
            gOSAL_C674_InterruptVectorTable.svcHandler = fxn;
            break;
        }
        default:
        {
            /* This is a catch-all condition. Control should never come here because this
             * violates the enumeration range. */
            break;
        }
    }
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the hook for the R4F CPU
 *      exception handler.
 *
 *  @param[in] type
 *      Exception Type for which the hook is to be deleted
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_C674_Interrupt_delExceptionHook (OSAL_C674_Exception type)
{
    switch (type)
    {
        case OSAL_C674_Exception_PREFETCH_ABORT:
        {
            gOSAL_C674_InterruptVectorTable.prefetchAbortHandler = OSAL_C674_Interrupt_defaultHandler;
            break;
        }
        case OSAL_C674_Exception_DATA_ABORT:
        {
            gOSAL_C674_InterruptVectorTable.dataAbortHandler = OSAL_C674_Interrupt_defaultHandler;
            break;
        }
        case OSAL_C674_Exception_UNDEFINED_INSTR:
        {
            gOSAL_C674_InterruptVectorTable.undefInstrHandler = OSAL_C674_Interrupt_defaultHandler;
            break;
        }
        case OSAL_C674_Exception_SVC:
        {
            gOSAL_C674_InterruptVectorTable.svcHandler = OSAL_C674_Interrupt_defaultHandler;
            break;
        }
        default:
        {
            /* This is a catch-all condition. Control should never come here because this
             * violates the enumeration range. */
            break;
        }
    }
}
#endif

/**
 *  @b Description
 *  @n
 *      In a working system the ISR is owned by the application
 *      However when a diagnostic is executed it would require a
 *      callback ("Hook") to be invoked by the application ISR. This
 *      is needed since the diagnostic will perform the necessary
 *      book keeping in the context of this function. Hooks are
 *      added during the start of the diagnostic execution.
 *
 *      The function is used to add the specified hook function
 *      on the reception of the specified interrupt.
 *
 *  @param[in] eventId
 *      This is the C674 Event Identifier associated with
 *      the interrupt.
 *  @param[in] hookFxn
 *      Hook function to be installed
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_Interrupt_addHook (uint8_t eventId, OSAL_HookFxn hookFxn)
{
    /* For TI-RTOS it uses mmwave SDK Hwi OSAL HwiP layer */
    HwiP_Params     hwiParams;
    uint8_t idx;

    /* find the empty Hwi Handle map */
    for(idx=0; idx<MAX_HWI_HANDLE; idx++)
    {
        if((gHwiHandleMap[idx].hwiHandle == NULL) && (gHwiHandleMap[idx].irq == 0))
            break;
    }

    HwiP_Params_init(&hwiParams);
    hwiParams.name = "SDL_IRQ_HOOK";
    //TODO JIT: May need to set enableInt to hwiParams
    /* HwiP_Type_IRQ or HwiP_Type_FIQ; */
    hwiParams.type = (HwiP_Type)HwiP_Type_IRQ;
    gHwiHandleMap[idx].irq = eventId;
    gHwiHandleMap[idx].hwiHandle = HwiP_create(eventId, (HwiP_Fxn)hookFxn, &hwiParams);
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the hook function for the
 *      specified interrupt.
 *
 *  @param[in] eventId
 *      This is the C674 Event Identifier associated with
 *      the interrupt.
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_Interrupt_delHook (uint8_t eventId)
{
    /* For TI-RTOS it uses mmwave SDK Hwi OSAL HwiP layer */
    uint8_t idx;

    /* find the empty Hwi Handle map */
    for(idx=0; idx < MAX_HWI_HANDLE; idx++)
    {
        if(gHwiHandleMap[idx].irq == eventId)
            break;
    }

    HwiP_delete(((HwiP_Handle)gHwiHandleMap[idx].hwiHandle));

    gHwiHandleMap[idx].hwiHandle = NULL;
    gHwiHandleMap[idx].irq       = 0;
}


/**
 *  @b Description
 *  @n
 *      In a working system the NMI is owned by the application
 *      However when a diagnostic is executed it would require a
 *      callback ("Hook") to be invoked by the application NMI handler. This
 *      is needed since the diagnostic will perform the necessary
 *      book keeping in the context of this function. Hooks are
 *      added during the start of the diagnostic execution.
 *
 *      The function is used to add the specified hook function
 *      on the reception of the specified interrupt.
 *
 *  @param[in] hookFxn
 *      Hook function to be installed
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_NMI_addHook (OSAL_HookFxn hookFxn)
{



    /* Add the NMI hook needed by the diagnostic */
   // gC674InterruptVectorTable.nmiHandler = hookFxn;
}

/**
 *  @b Description
 *  @n
 *      The function is used to delete the NMI hook function.
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_NMI_delHook ()
{
    /* Remove the hook by adding back the default handler */
    //gC674InterruptVectorTable.nmiHandler = OSAL_C674_Interrupt_defaultHandler;
}

/**
 *  @b Description
 *  @n
 *      The function has been provided as a utility function
 *      to initialize the C674 Interrupt Controller (INTC).
 *      Interrupt handling can only be done after this function
 *      has been invoked. This function is only invoked by the
 *      unit tests to verify diagnostic functionality. System
 *      executing with an embedded operating system would
 *      have this functionality as a part of the operating
 *      system startup procedure.
 *
 *  \ingroup OSAL_INTERRUPT
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_Interrupt_init (void)
{
    Hwi_Params  params;
    uint32_t    i;

    /* For TI-RTOS it uses mmwave SDK Hwi OSAL HwiP clear interrupt */
    int index;
    for ( index = 0U; index <= 126U; index++)
    {
        HwiP_clearInterrupt(index);
    }

    /* setup Event Combiner */
    Hwi_Params_init(&params);
    params.enableInt = TRUE;
    for (i = 0; i < 4; i++)
    {
        params.arg      = i;
        params.eventId  = i;
        if (Hwi_create(4 + i, &EventCombiner_dispatch, &params, NULL) == NULL)
        {
            System_printf("failed to create Hwi interrupt %d\n",4 + i);
        }
    }

}


