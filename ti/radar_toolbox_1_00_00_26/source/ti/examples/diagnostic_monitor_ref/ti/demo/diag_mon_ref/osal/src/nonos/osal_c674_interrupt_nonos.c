/*
 *   @file  osal_c674_interrupt_nonos.c
 *
 *   @brief
 *          Implementation for C674x Non-OS interrupt management Functionality
 *          File is mainly used for Safety Diagnostic Library (SDL).
 *
 *  \par
 *  NOTE:
 *      (C) Copyright 2020 Texas Instruments, Inc.
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
#include <c6x.h>

/* mmWave SDL Include Files */
#include <ti/csl/csl.h>
#include "osal/osal.h"

/**************************************************************************
 **************************** Local Declarations **************************
 **************************************************************************/

static void OSAL_C674_Interrupt_defaultHandler (void);
static void OSAL_C674_Interrupt_executeEventCombiner (uint8_t eventGroup);
static void OSAL_C674_Interrupt_EventCombiner0Handler(void);
static void OSAL_C674_Interrupt_EventCombiner1Handler(void);
static void OSAL_C674_Interrupt_EventCombiner2Handler(void);
static void OSAL_C674_Interrupt_EventCombiner3Handler(void);

/**************************************************************************
 **************************** Extern Declarations *************************
 **************************************************************************/

/* These functions are invoked from the C674 Interrupt Vector Table. */
extern void OSAL_C674_Interrupt_NMIHandler (void);
extern void OSAL_C674_Interrupt_reserved (void);
extern void OSAL_C674_Interrupt_int4Handler (void);
extern void OSAL_C674_Interrupt_int5Handler (void);
extern void OSAL_C674_Interrupt_int6Handler (void);
extern void OSAL_C674_Interrupt_int7Handler (void);
extern void OSAL_C674_Interrupt_int8Handler (void);
extern void OSAL_C674_Interrupt_int9Handler (void);
extern void OSAL_C674_Interrupt_int10Handler (void);
extern void OSAL_C674_Interrupt_int11Handler (void);
extern void OSAL_C674_Interrupt_int12Handler (void);
extern void OSAL_C674_Interrupt_int13Handler (void);
extern void OSAL_C674_Interrupt_int14Handler (void);
extern void OSAL_C674_Interrupt_int15Handler (void);

/* This function is used only by the Test code to initialize the OSAL C674
 * Interrupt module. This is the reason why it has not been added to the
 * osal.h. The osal.h will only have a list of all the exported functions
 * used by the diagnostics and which need to be ported by the customer. */
extern void OSAL_C674_Interrupt_init (void);

/**************************************************************************
 ***************************** Local Structures ***************************
 **************************************************************************/

/**
 * @brief
 *  Interrupt Vector Table
 *
 * @details
 *  The structure is used to hold information of the registered
 *  interrupt handlers which are invoked on the reception of the
 *  interrupt.
 *
 *  \ingroup OSAL_INTERNAL_STRUCTURES
 */
typedef struct OSAL_C674_Interrupt_VectorTable_t
{
    /**
     * @brief   NMI Handler:
     */
    OSAL_HookFxn    nmiHandler;

    /**
     * @brief   C674 CPU Interrupt Handler:
     */
    OSAL_HookFxn    intHandler[12];
}OSAL_C674_Interrupt_VectorTable;

/**
 * @brief
 *  CPU Event Id ISR Table
 *
 * @details
 *  The structure is used to hold information of the registered
 *  interrupt handlers which are invoked on the reception of the
 *  specified CPU Event.
 *
 *  \ingroup OSAL_INTERNAL_STRUCTURES
 */
typedef struct OSAL_C674_Interrupt_EventTable_t
{
    /**
     * @brief   The C674 can handle 128 system events.
     */
    OSAL_HookFxn    eventHandler[128];
}OSAL_C674_Interrupt_EventTable;

/**************************************************************************
 **************************** Global Declarations *************************
 **************************************************************************/

/**
 * @brief   Interrupt Vector Table:
 *  INT4 -> Event Combiner 0 is handled here
 *  INT5 -> Event Combiner 1 is handled here
 *  INT6 -> Event Combiner 2 is handled here
 *  INT7 -> Event Combiner 3 is handled here
 */
OSAL_C674_Interrupt_VectorTable  gC674InterruptVectorTable =
{
    OSAL_C674_Interrupt_defaultHandler,             /* NMI Handler:               */
    {
        OSAL_C674_Interrupt_EventCombiner0Handler,  /* CPU Interrupt 4 Handler:   */
        OSAL_C674_Interrupt_EventCombiner1Handler,  /* CPU Interrupt 5 Handler:   */
        OSAL_C674_Interrupt_EventCombiner2Handler,  /* CPU Interrupt 6 Handler:   */
        OSAL_C674_Interrupt_EventCombiner3Handler,  /* CPU Interrupt 7 Handler:   */
        OSAL_C674_Interrupt_defaultHandler,         /* CPU Interrupt 8 Handler:   */
        OSAL_C674_Interrupt_defaultHandler,         /* CPU Interrupt 9 Handler:   */
        OSAL_C674_Interrupt_defaultHandler,         /* CPU Interrupt 10 Handler:  */
        OSAL_C674_Interrupt_defaultHandler,         /* CPU Interrupt 11 Handler:  */
        OSAL_C674_Interrupt_defaultHandler,         /* CPU Interrupt 12 Handler:  */
        OSAL_C674_Interrupt_defaultHandler,         /* CPU Interrupt 13 Handler:  */
        OSAL_C674_Interrupt_defaultHandler,         /* CPU Interrupt 14 Handler:  */
        OSAL_C674_Interrupt_defaultHandler          /* CPU Interrupt 15 Handler:  */
    }
};

/**
 * @brief   Event Table:
 */
OSAL_C674_Interrupt_EventTable   gOSAL_C674_EventTable;

/**************************************************************************
 ******************************** Functions *******************************
 **************************************************************************/

/**
 *  @b Description
 *  @n
 *      The function is used to clear the event mask. Event masks allow upto
 *      32 events to be combined together into a single event output that
 *      is used as a CPU interrupt.
 *
 *  @param[in] ptrINTCRegs
 *      Pointer to the INTC Register
 *  @param[in] eventId
 *      Event Identifier to be combined
 *
 *  \ingroup CSL_C674_FUNCTION
 *
 *  @retval
 *      Not Applicable.
 *
 *  @note
 *   - C674 documentation states that there are 124 events
 */
void CSL_INTC_clearEventCombiner (CSL_INTCRegs* ptrINTCRegs, uint8_t eventId)
{
    uint8_t     reg;
    uint8_t     bit;

    /* There are 32 events per registers: We need to calculate the
     * register index which needs to be configured. */
    reg = eventId / 32U;

    /* Compute the bit position in the register: */
    bit = eventId % 32U;

    /* Clear the register: */
    ptrINTCRegs->EVTMASK[reg] = CSL_insert8 (ptrINTCRegs->EVTMASK[reg], bit, bit, 1U);
}
/**
 *  @b Description
 *  @n
 *      This is the registered default handler which is installed
 *      for all CPU Interrupts & Events. The function will
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

/**
 *  @b Description
 *  @n
 *      The function cycles through all the events which have been
 *      grouped together through the event combiner. The function
 *      will decode the pending event and will invoke the registered
 *      handler.
 *
 *  @param[in] eventGroup
 *      Event Group
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 */
static void OSAL_C674_Interrupt_executeEventCombiner (uint8_t eventGroup)
{
    uint8_t         baseEventIndex;
    uint32_t        eventMask;
    uint8_t         index;
    CSL_INTCRegs*   ptrINTCRegs;

    /* Get the base address of the INTC Module: */
    ptrINTCRegs = CSL_INTC_getBaseAddress();

    /* Compute the base event table index:
     *   - Event Group 0: Events 4  to 31
     *   - Event Group 1: Events 32 to 63
     *   - Event Group 2: Events 64 to 95
     *   - Event Group 3: Events 96 to 127 */
    if (eventGroup == 0U)
    {
        baseEventIndex = 0U;
    }
    else if (eventGroup == 1U)
    {
        baseEventIndex = 32U;
    }
    else if (eventGroup == 2U)
    {
        baseEventIndex = 64U;
    }
    else
    {
        baseEventIndex = 96U;
    }

    /* Get the pending events: */
    eventMask = CSL_INTC_getPendingEventMask (ptrINTCRegs, eventGroup);

    /* Cycle through and invoke the corresponding event handler: */
    while (eventMask != 0U)
    {
        /* Cycle through each bit: Check if it is enabled? */
        for (index = 0U; index < 32U; index++)
        {
            /* Is the mask set? */
            if (CSL_extract8 (eventMask, index, index) == 1U)
            {
                /* YES: Clear the pending event: */
                CSL_INTC_clearEvent (ptrINTCRegs, (baseEventIndex + index));

                /* Invoke the event function */
                gOSAL_C674_EventTable.eventHandler[baseEventIndex + index]();
            }
            else
            {
                /* NO: Continue looking for the event */
            }
        }
        /* Get the pending events: */
        eventMask = CSL_INTC_getPendingEventMask (ptrINTCRegs, eventGroup);
    }
}

/**
 *  @b Description
 *  @n
 *      This function is used to process events associated
 *      with the Event Group 0
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 */
static void OSAL_C674_Interrupt_EventCombiner0Handler(void)
{
    /* Process the events in Event Group 0 */
    OSAL_C674_Interrupt_executeEventCombiner (0U);
}

/**
 *  @b Description
 *  @n
 *      This function is used to process events associated
 *      with the Event Group 1
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 */
static void OSAL_C674_Interrupt_EventCombiner1Handler(void)
{
    /* Process the events in Event Group 1 */
    OSAL_C674_Interrupt_executeEventCombiner (1U);
}

/**
 *  @b Description
 *  @n
 *      This function is used to process events associated
 *      with the Event Group 2
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 */
static void OSAL_C674_Interrupt_EventCombiner2Handler(void)
{
    /* Process the events in Event Group 2 */
    OSAL_C674_Interrupt_executeEventCombiner (2U);
}

/**
 *  @b Description
 *  @n
 *      This function is used to process events associated
 *      with the Event Group 3
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 */
static void OSAL_C674_Interrupt_EventCombiner3Handler(void)
{
    /* Process the events in Event Group 3 */
    OSAL_C674_Interrupt_executeEventCombiner (3U);
}

/**
 *  @b Description
 *  @n
 *      This is the C674 NMI Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma NMI_INTERRUPT (OSAL_C674_Interrupt_NMIHandler)
void OSAL_C674_Interrupt_NMIHandler (void)
{
    gC674InterruptVectorTable.nmiHandler();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 reserved handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_reserved)
void OSAL_C674_Interrupt_reserved (void)
{
    volatile uint32_t deadloop = 1U;
    while (deadloop == 1U)
    {
    }
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 4 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int4Handler)
void OSAL_C674_Interrupt_int4Handler (void)
{
    gC674InterruptVectorTable.intHandler[0]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 5 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int5Handler)
void OSAL_C674_Interrupt_int5Handler (void)
{
    gC674InterruptVectorTable.intHandler[1]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 6 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int6Handler)
void OSAL_C674_Interrupt_int6Handler (void)
{
    gC674InterruptVectorTable.intHandler[2]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 7 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int7Handler)
void OSAL_C674_Interrupt_int7Handler (void)
{
    gC674InterruptVectorTable.intHandler[3]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 8 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int8Handler)
void OSAL_C674_Interrupt_int8Handler (void)
{
    gC674InterruptVectorTable.intHandler[4]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 9 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int9Handler)
void OSAL_C674_Interrupt_int9Handler (void)
{
    gC674InterruptVectorTable.intHandler[5]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 10 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int10Handler)
void OSAL_C674_Interrupt_int10Handler (void)
{
    gC674InterruptVectorTable.intHandler[6]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 11 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int11Handler)
void OSAL_C674_Interrupt_int11Handler (void)
{
    gC674InterruptVectorTable.intHandler[7]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 12 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int12Handler)
void OSAL_C674_Interrupt_int12Handler (void)
{
    gC674InterruptVectorTable.intHandler[8]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 13 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int13Handler)
void OSAL_C674_Interrupt_int13Handler (void)
{
    gC674InterruptVectorTable.intHandler[9]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 14 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int14Handler)
void OSAL_C674_Interrupt_int14Handler (void)
{
    gC674InterruptVectorTable.intHandler[10]();
}

/**
 *  @b Description
 *  @n
 *      This is the C674 CPU Interrupt 15 Handler which is installed in
 *      the C674 CPU Interrupt Table. This is invoked from
 *      the CPU Interrupt Dispatcher code and so these
 *      functions need to be written with the INTERRUPT
 *      keyword
 *
 *  \ingroup OSAL_INTERNAL_FUNCTIONS
 *
 *  @retval
 *      Not Applicable.
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
#pragma INTERRUPT (OSAL_C674_Interrupt_int15Handler)
void OSAL_C674_Interrupt_int15Handler (void)
{
    gC674InterruptVectorTable.intHandler[11]();
}

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
    CSL_INTCRegs*   ptrINTCRegs;

    /* Sanity Checking:
     * - Event Id 0 to 3 are reserved for the event combiner
     * - Maximum number of events supported are 128.
     * This is as per the C674 documentation */
    if ((eventId >= 4U) && (eventId < 128U))
    {
        /* Get the base address of the INTC Module: */
        ptrINTCRegs = CSL_INTC_getBaseAddress();

        /* Register the ISR into the event table. */
        gOSAL_C674_EventTable.eventHandler[eventId] = hookFxn;

        /* Enable the event to be combined: */
        CSL_INTC_setEventCombiner (ptrINTCRegs, eventId);
    }
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
    /* Sanity Checking:
     * - Event Id 0 to 3 are reserved for the event combiner
     * - Maximum number of events supported are 128.
     * This is as per the C674 documentation */
    if ((eventId >= 4U) && (eventId < 128U))
    {
        CSL_INTC_clearEventCombiner(CSL_INTC_getBaseAddress(), eventId);

        /* Remove the hook by adding back the default handler */
        gOSAL_C674_EventTable.eventHandler[eventId] = OSAL_C674_Interrupt_defaultHandler;
    }
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
    gC674InterruptVectorTable.nmiHandler = hookFxn;
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
    gC674InterruptVectorTable.nmiHandler = OSAL_C674_Interrupt_defaultHandler;
}

/**
 *  @b Description
 *  @n
 *      The function is used to clear a system event.
 *
 *  @param[in] eventId
 *      Event Identifier to be set
 *
 *  @retval
 *      Not Applicable.
 *
 *  @note
 *   - C674 documentation states that there are 124 events
 */
void OSAL_C674_IntEventClear(uint8_t eventId)
{
    CSL_INTCRegs*   ptrINTCRegs;

    /* Get the base address of the INTC Module: */
    ptrINTCRegs = CSL_INTC_getBaseAddress();

    /* Clear any pending event: */
    CSL_INTC_clearEvent (ptrINTCRegs, eventId);
}

/**
 *  @b Description
 *  @n
 *      The function is used to set a system event.
 *
 *  @param[in] eventId
 *      Event Identifier to be set
 *
 *  @retval
 *      Not Applicable.
 *
 *  @note
 *   - C674 documentation states that there are 124 events
 */
void OSAL_C674_IntEventSet(uint8_t eventId)
{
    CSL_INTCRegs*   ptrINTCRegs;
    uint8_t     reg;
    uint8_t     bit;
    /* Get the base address of the INTC Module: */
    ptrINTCRegs = CSL_INTC_getBaseAddress();

    /* There are 32 events per registers: We need to calculate the
     * register index which needs to be configured. */
    reg = eventId / 32U;

    /* Compute the bit position in the register: */
    bit = eventId % 32U;

    /* Clear the register: */
    ptrINTCRegs->EVTSET[reg] = CSL_insert8 (ptrINTCRegs->EVTSET[reg], bit, bit, 1U);
}

/**
 *  @b Description
 *  @n
 *      The function is used to set a event combiner for eventID.
 *
 *  @param[in] eventId
 *      Event Identifier to be set
 *
 *  @retval
 *      Not Applicable.
 *
 *  @note
 *   - C674 documentation states that there are 124 events
 */
void OSAL_C674_SetEventCombiner(uint8_t eventId)
{
    CSL_INTCRegs*   ptrINTCRegs;

    /* Sanity Checking:
     * - Event Id 0 to 3 are reserved for the event combiner
     * - Maximum number of events supported are 128.
     * This is as per the C674 documentation */
    if ((eventId >= 4U) && (eventId < 128U))
    {
        /* Get the base address of the INTC Module: */
        ptrINTCRegs = CSL_INTC_getBaseAddress();

        /* Enable the event to be combined: */
        CSL_INTC_setEventCombiner (ptrINTCRegs, eventId);
    }
}

/**
 *  @b Description
 *  @n
 *      The function is used to remove eventId from event combiner
 *
 *  @param[in] eventId
 *      Event Identifier to be set
 *
 *  @retval
 *      Not Applicable.
 *
 *  @note
 *   - C674 documentation states that there are 124 events
 */
void OSAL_C674_ClearEventCombiner(uint8_t eventId)
{
    CSL_INTCRegs*   ptrINTCRegs;

    /* Sanity Checking:
     * - Event Id 0 to 3 are reserved for the event combiner
     * - Maximum number of events supported are 128.
     * This is as per the C674 documentation */
    if ((eventId >= 4U) && (eventId < 128U))
    {
        /* Get the base address of the INTC Module: */
        ptrINTCRegs = CSL_INTC_getBaseAddress();

        /* Clear the event to be combined: */
        CSL_INTC_clearEventCombiner (ptrINTCRegs, eventId);
    }
}

/**
 * \function IntGlobalEnable
 *
 * \brief    Enable DSP CPU interrupts globally.
 *
 * \param    None
 *
 * \return   None
 */
void OSAL_C674_IntGlobalEnable (void)
{
    _enable_interrupts();
}

/**
 * \function IntGlobalDisable
 *
 * \brief    Disable DSP CPU interrupts globally
 *
 * \param    None
 *
 * \return   DSP CPU interrupt state
 */
uint32_t OSAL_C674_IntGlobalDisable (void)
{
    return (uint32_t)(_disable_interrupts());
}

/**
 * \function IntGlobalRestore
 *
 * \brief    Restore the DSP CPU interrupt state
 *
 * \param    Previous DSP CPU interrupt state
 *
 * \return   None
 */
void OSAL_C674_IntGlobalRestore (uint32_t restoreValue)
{
    _restore_interrupts(restoreValue);
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
    CSL_INTCRegs*   ptrINTCRegs;
    uint8_t         index;

    /* Get the base address of the INTC Module: */
    ptrINTCRegs = CSL_INTC_getBaseAddress();

    /* Cycle through and clear out any pending events: */
    for (index = 0U; index < 128U; index++)
    {
        /* Clear any pending event: */
        CSL_INTC_clearEvent (ptrINTCRegs, index);

        /* Mask all the interrupt, and enabled based on application request. */
        CSL_INTC_clearEventCombiner(ptrINTCRegs, index);

        /* Default event handler are installed: */
        gOSAL_C674_EventTable.eventHandler[index] = OSAL_C674_Interrupt_defaultHandler;
    }

    /* Clear off all the maskable interrupts: */
    ICR = 0xffff;


    /* Event combiner:
     * - Setup Event Group 0 error to CPU Interrupt 4
     * - Setup Event Group 1 error to CPU Interrupt 5
     * - Setup Event Group 2 error to CPU Interrupt 6
     * - Setup Event Group 3 error to CPU Interrupt 7
     *
     * This is inline with the CPU Event Handlers which have
     * been installed */
    CSL_INTC_setInterruptMux (ptrINTCRegs, 4U, 0x0);
    CSL_INTC_setInterruptMux (ptrINTCRegs, 5U, 0x1);
    CSL_INTC_setInterruptMux (ptrINTCRegs, 6U, 0x2);
    CSL_INTC_setInterruptMux (ptrINTCRegs, 7U, 0x3);

    /***************************************************************
     * C674 Documentation states that in order to process a maskable
     * interrupt the following conditions needs to be met:-
     * 1. The GIE bit in the CSR register should be set
     * 2. The NMIE bit in the IER should be set.
     * 3. The corresponding IE bit in the IER should be set
     ***************************************************************/

    /* Step1: Enable the global interrupts: */
    CSR = CSL_insert8 (CSR, 0U, 0U, 1U);

    /* Step2: Set the NMIE bit in the IER */
#if 1
    IER = CSL_insert8 (IER, 1U, 1U, 1U);
#else
    IER = 1<<1;//CSL_insert8 (IER, 1U, 1U, 1U); //TODO JIT 09022021
#endif

    /* Step3: Enable all the interrupts */
#if 0
    for (index = 4U; index <= 15U; index++)
#else
    for (index = 4U; index <= 7U; index++)
#endif
    {
        IER = CSL_insert8 (IER, index, index, 1U);//TODO INTC: NOT THERE
    }
    return;
}

