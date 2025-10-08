/*
 * OSAL Header File
 *
 * This is a test file which has been added to only verify build
 * functionality
 *
 * Copyright (C) 2019 Texas Instruments Incorporated - http://www.ti.com/
 * ALL RIGHTS RESERVED
 *
 */

#ifndef OSAL_H
#define OSAL_H

/**
@defgroup OSAL                  Operation System Porting Layer
@ingroup MMWAVE_SDL
@brief
*   Operating System Porting Layer
*
@details
*   The OSAL layer has been designed to ensure that the diagnostics
*   can be ported and used within an operating system with minimal
*   changes.
*
*   The mmWave SDL package has been ported and tested on a bare metal
*   i.e. no operating system. However if an embedded system RTOS is being
*   used there are certain aspects which need to be accounted for.
*
*   * Entry Point:\n
*   * Interrupt/Exception Managment:\n
*   * Cycle Profiling:\n
*   * ESM:\n
*/
/**
@defgroup OSAL_CYCLE_PROFILING                  Cycle Profiling
@ingroup OSAL
@details
* The cycle profiling functionality allows CPU architecture specific counters
* to be used to get the cycle counts for benchmarking the diagnostic functions.
* This is also used to implement the timeout facilities which ensure that
* diagnostic functions on error conditions do not get stuck in endless while
* loops
*/
/**
@defgroup OSAL_INTERRUPT                        Interrupt Management
@ingroup OSAL
@details
* The interrupt layer is reponsible for the installation of the CPU architecture
* interrupt vector table. The interrupt layer allows applications to install
* handlers for interrupts & exceptions. It ensures that the exception entry & exit
* is as per the CPU specifications.
*/
/**
@defgroup OSAL_ESM_DRV_INTERFACE                ESM Driver Interface
@ingroup OSAL
@details
* The ESM driver is not a part of the operating system. However all applications
* executing in the system will have an ESM driver executing in the system. The
* mmWave SDL diagnostic functions will be generating ESM errors in the system
* and the interface defined here will allow the SDL layer to work with the
* existing drivers to properly handle the diagnostic errors.
*/
/**
@defgroup OSAL_STRUCTURES                       Exported Data Structures
@ingroup OSAL
@details
* The section has a list of all the data structures which are available
* to the applications.
*/
/**
@defgroup OSAL_INTERNAL_FUNCTIONS               Internal Functions
@ingroup OSAL
@details
* The section has a list of the internal functions which are not exposed
* and should not be used by the applications.
*/
/**
@defgroup OSAL_INTERNAL_STRUCTURES              Internal Structures
@ingroup OSAL
@details
* The section has a list of the internal data structures which are not exposed
* and should not be used by the applications.
*/

/**************************************************************************
 *************************** Include Files ********************************
 **************************************************************************/
#include <stdint.h>
#include <stdio.h>

/* mmWave SDL Include files: */
#include <ti/csl/csl.h>

#ifdef __cplusplus
extern "C" {
#endif

/** @addtogroup OSAL_STRUCTURES
 @{ */

/**
 * @brief
 *  Hook Function
 *
 * @details
 *  This is the generic hook function. Hooks can be registered
 *  and will be invoked on the reception of the specified
 *  interrupt/CPU exception.
 */
typedef void (*OSAL_HookFxn) (void);

/**
 * @brief
 *  R4F Exception
 *
 * @details
 *  The enumeration describes the R4F Exceptions. These are as per the
 *  ARM R4F documentation.
 */
typedef enum OSAL_R4F_Exception_e
{
    /**
     * @brief   Prefetch Abort Exception
     */
    OSAL_R4F_Exception_PREFETCH_ABORT = 0x1,

    /**
     * @brief   Data Abort Exception
     */
    OSAL_R4F_Exception_DATA_ABORT,

    /**
     * @brief   Supervisor Call
     */
    OSAL_R4F_Exception_SVC,

    /**
     * @brief   Undefined Instruction Exception
     */
    OSAL_R4F_Exception_UNDEFINED_INSTR
}OSAL_R4F_Exception;

/**
@}
*/

/**************************************************************************
 ************************* Extern Declarations ****************************
 **************************************************************************/

/*************************************************************************
 * Interrupt Management Module Exported API for the R4F
 ************************************************************************/
extern void OSAL_R4F_Interrupt_addHook (uint8_t irq, uint8_t fiqType, OSAL_HookFxn hookFxn);
extern void OSAL_R4F_Interrupt_delHook (uint8_t irq);
extern void OSAL_R4F_Interrupt_addExceptionHook (OSAL_R4F_Exception type, OSAL_HookFxn fxn);
extern void OSAL_R4F_Interrupt_delExceptionHook (OSAL_R4F_Exception type);

/*************************************************************************
 * Cycle Profiling Module Exported API for the R4F
 ************************************************************************/
extern uint32_t OSAL_R4F_CycleProfiler_getCount (void);

/*************************************************************************
 * ESM Driver Interface Module Exported API for the R4F
 ************************************************************************/
extern void OSAL_R4F_ESMDrv_addHook (const CSL_ESM_ErrorChannel* ptrErrorChannel, OSAL_HookFxn fxn);
extern void OSAL_R4F_ESMDrv_delHook (const CSL_ESM_ErrorChannel* ptrErrorChannel);

/* R4F handling of DSS ESM */
extern uint8_t OSAL_R4F_DSSESMDrv_getInterrupt (CSL_ESM_ErrorChannel* ptrDSSErrorChannel);
extern void OSAL_R4F_DSSESMDrv_enableInterrupt (const CSL_ESM_ErrorChannel* ptrDSSErrorChannel);
extern void OSAL_R4F_DSSESMDrv_disableInterrupt (const CSL_ESM_ErrorChannel* ptrDSSErrorChannel);
extern void OSAL_R4F_DSSESMDrv_clearInterrupt (const CSL_ESM_ErrorChannel* ptrDSSErrorChannel);

/*************************************************************************
 * Interrupt Management Module Exported API for the C674
 ************************************************************************/
extern void OSAL_C674_Interrupt_addHook (uint8_t eventId, OSAL_HookFxn hookFxn);
extern void OSAL_C674_Interrupt_delHook (uint8_t eventId);
extern void OSAL_C674_Interrupt_init (void);
extern void OSAL_C674_IntEventClear(uint8_t eventId);
extern uint32_t OSAL_C674_IntGlobalDisable (void);
extern void OSAL_C674_IntGlobalRestore (uint32_t restoreValue);
extern void OSAL_C674_SetEventCombiner(uint8_t eventId);
extern void OSAL_C674_ClearEventCombiner(uint8_t eventId);

/*************************************************************************
 * NMI Management Module Exported API for the C674
 ************************************************************************/
extern void OSAL_C674_NMI_addHook (OSAL_HookFxn hookFxn);
extern void OSAL_C674_NMI_delHook (void);

/*************************************************************************
 * Cycle Profiling Module Exported API for the C674
 ************************************************************************/
extern void OSAL_C674_CycleProfiler_init (void);
extern uint32_t OSAL_C674_CycleProfiler_getCount (void);


/*************************************************************************
 * ESM Driver Interface Module Exported API for the C674
 ************************************************************************/
extern void OSAL_C674_ESMDrv_init (void);
extern void OSAL_C674_ESMDrv_addHook (const CSL_ESM_ErrorChannel* ptrErrorChannel, OSAL_HookFxn fxn);
extern void OSAL_C674_ESMDrv_delHook (const CSL_ESM_ErrorChannel* ptrErrorChannel);

#ifdef __cplusplus
}
#endif

#endif /* OSAL_H */

