/*
 *   @file  osal_cycle_profiler_nonos.c
 *
 *   @brief
 *          This file implements the R4F & C674x Non-OS OSAL Cycle Profiler Functionality
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

#include "osal/osal.h"
/* mmWave SDL Include Files */
#include <ti/csl/csl.h>

/**************************************************************************
 **************************** Global Declarations *************************
 **************************************************************************/
#if defined (SUBSYS_MSS)
/**
 * @brief   Global variable for the Cycle Profiler MCB
 */
uint32_t gOSAL_CycleProfiler_counterId = 0U;
#endif

/**************************************************************************
 **************************** Extern Declarations *************************
 **************************************************************************/

/* This function is used only by the Test code to initialize the OSAL R4F
 * Cycle Profiling module. This is the reason why it has not been added to the
 * osal.h. The osal.h will only have a list of all the exported functions
 * used by the diagnostics and which need to be ported by the customer. */
extern void OSAL_R4F_CycleProfiler_init (uint32_t pmuCounterId);
extern void OSAL_C674_CycleProfiler_init (void);

/**************************************************************************
 ******************************** Functions *******************************
 **************************************************************************/
#if defined (SUBSYS_MSS)
/**
 *  @b Description
 *  @n
 *      The function is used to get the cycle count. On the
 *      R4F the cycle count is read from the PMU counter
 *      identifier which was specified when the module was
 *      initialized.
 *
 *  \ingroup OSAL_CYCLE_PROFILING
 *
 *  @retval
 *      Cycle count
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
uint32_t OSAL_R4F_CycleProfiler_getCount (void)
{
    /* Select the PMU counter and read the cycle count. */
    CSL_R4F_selectCounter (gOSAL_CycleProfiler_counterId);
    return CSL_R4F_getCycleCount();        
}

/**
 *  @b Description
 *  @n
 *      The function is used to initialize the R4F PMU counters. The
 *      function will setup the PMU Counter specified in the argument
 *      and will use this for all cycle profiling.
 *
 *      This function is provided as reference only code and is invoked
 *      only from the unit test code.
 *
 *  @param[in] pmuCounterId
 *      PMU counter identifier
 *
 *  \ingroup OSAL_CYCLE_PROFILING
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-28)
 */
void OSAL_R4F_CycleProfiler_init (uint32_t pmuCounterId)
{
    /* Sanity Checking: PMU Counter Id should be within range */
    if (pmuCounterId <= 2U)
    {
        /* Enable all the CPU Counters: */
        CSL_R4F_enableAllCounters();

        /* Reset all the counters */
        CSL_R4F_resetAllEventCounters();
        CSL_R4F_resetCycleCounter();

        /* Enable the cycle count: */
        CSL_R4F_enableCycleCount ();

        /* Enable the Cycle Profiler: */
        gOSAL_CycleProfiler_counterId = pmuCounterId;

        /* Enable the counter: */
        CSL_R4F_selectCounter (gOSAL_CycleProfiler_counterId);
        CSL_R4F_enableCounter (gOSAL_CycleProfiler_counterId);
    }
}
#endif

#if defined (SUBSYS_DSS)
#include <c6x.h>
/**
 *  @b Description
 *  @n
 *      The function is used to get the cycle count. On the
 *      C674 the cycle counter is the snapshot of the TSCL
 *      register.
 *
 *  \ingroup OSAL_CYCLE_PROFILING
 *
 *  @retval
 *      Cycle count
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
uint32_t OSAL_C674_CycleProfiler_getCount (void)
{
    return TSCL;
}

/**
 *  @b Description
 *  @n
 *      The function is used to initialize the cycle profiler on the C674
 *
 *      This function is provided as reference only code and is invoked
 *      only from the unit test code.
 *
 *  \ingroup OSAL_CYCLE_PROFILING
 *
 *  @retval
 *      Not applicable
 *
 * Design: did_mmwave_osal
 * Architecture: aid_mmwave_sdl_osal
 * Requirement: REQ_TAG(MMWSDL-38)
 */
void OSAL_C674_CycleProfiler_init (void)
{
    /* Enable the C674 Timestamp counter: */
    TSCL = 0;
}
#endif
