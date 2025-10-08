/**
 *   @file  diag_mon_mss.h
 *
 *   @brief
 *      This is the main header file for Diagnostic & Monitor Demo
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
#ifndef DIAG_MON_MSS_H
#define DIAG_MON_MSS_H

#include <ti/sysbios/knl/Semaphore.h>
#include <ti/sysbios/knl/Task.h>

#include <ti/common/mmwave_error.h>
#include <ti/drivers/osal/DebugP.h>
#include <ti/drivers/soc/soc.h>
#include <ti/drivers/uart/UART.h>
#include <ti/drivers/gpio/gpio.h>
#include <ti/drivers/mailbox/mailbox.h>

#include "common/diag_mon_output.h"
#include "common/diag_mon_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/* definition for xWR1843 device variant */
#ifdef SOC_XWR18XX
#define MMW_SOC_PIN_UART1_TX        SOC_XWR18XX_PINN5_PADBE
#define MMW_SOC_PIN_UART1_TX_FUNC   SOC_XWR18XX_PINN5_PADBE_MSS_UARTA_TX
#define MMW_SOC_PIN_UART1_RX        SOC_XWR18XX_PINN4_PADBD
#define MMW_SOC_PIN_UART1_RX_FUNC   SOC_XWR18XX_PINN4_PADBD_MSS_UARTA_RX
#define MMW_SOC_PIN_UART3_TX        SOC_XWR18XX_PINF14_PADAJ
#define MMW_SOC_PIN_UART3_TX_FUNC   SOC_XWR18XX_PINF14_PADAJ_MSS_UARTB_TX
#define MMW_SOC_PIN_GPIO2           SOC_XWR18XX_PINK13_PADAZ
#define MMW_SOC_PIN_GPIO2_FUNC      SOC_XWR18XX_PINK13_PADAZ_GPIO_2
#define MMW_SOC_PIN_GPIO2_OUT       SOC_XWR18XX_GPIO_2
#define MMW_MMWAVELINK_DEVICE_MAP   RL_AR_DEVICETYPE_18XX
#define MMW_MSS_FRAME_START_INT     SOC_XWR18XX_MSS_FRAME_START_INT
#endif
/* definition for xWR1642 device variant */
#ifdef SOC_XWR16XX
#define MMW_SOC_PIN_UART1_TX        SOC_XWR16XX_PINN5_PADBE
#define MMW_SOC_PIN_UART1_TX_FUNC   SOC_XWR16XX_PINN5_PADBE_MSS_UARTA_TX
#define MMW_SOC_PIN_UART1_RX        SOC_XWR16XX_PINN4_PADBD
#define MMW_SOC_PIN_UART1_RX_FUNC   SOC_XWR16XX_PINN4_PADBD_MSS_UARTA_RX
#define MMW_SOC_PIN_UART3_TX        SOC_XWR16XX_PINF14_PADAJ
#define MMW_SOC_PIN_UART3_TX_FUNC   SOC_XWR16XX_PINF14_PADAJ_MSS_UARTB_TX
#define MMW_SOC_PIN_GPIO2           SOC_XWR16XX_PINK13_PADAZ
#define MMW_SOC_PIN_GPIO2_FUNC      SOC_XWR16XX_PINK13_PADAZ_GPIO_2
#define MMW_SOC_PIN_GPIO2_OUT       SOC_XWR16XX_GPIO_2
#define MMW_MMWAVELINK_DEVICE_MAP   RL_AR_DEVICETYPE_16XX
#define MMW_MSS_FRAME_START_INT     SOC_XWR16XX_MSS_FRAME_START_INT
#endif
/* definition for xWR1443 device variant */
#ifdef SOC_XWR14XX
#define MMW_SOC_PIN_UART1_TX        SOC_XWR14XX_PINN6_PADBE
#define MMW_SOC_PIN_UART1_TX_FUNC   SOC_XWR14XX_PINN6_PADBE_MSS_UARTA_TX
#define MMW_SOC_PIN_UART1_RX        SOC_XWR14XX_PINN5_PADBD
#define MMW_SOC_PIN_UART1_RX_FUNC   SOC_XWR14XX_PINN5_PADBD_MSS_UARTA_RX
#define MMW_SOC_PIN_UART3_TX        SOC_XWR14XX_PINN6_PADBE_MSS_UARTB_TX
#define MMW_SOC_PIN_UART3_TX_FUNC   SOC_XWR14XX_PINN6_PADBE_MSS_UARTB_TX
#define MMW_SOC_PIN_GPIO2           SOC_XWR14XX_PINN13_PADAZ
#define MMW_SOC_PIN_GPIO2_FUNC      SOC_XWR14XX_PINN13_PADAZ_GPIO_2
#define MMW_SOC_PIN_GPIO2_OUT       SOC_XWR14XX_GPIO_2
#define MMW_MMWAVELINK_DEVICE_MAP   RL_AR_DEVICETYPE_14XX
#define MMW_MSS_FRAME_START_INT     SOC_XWR14XX_BSS_FRAME_START_INT
#endif
/* definition for xWR684x device variant */
#ifdef SOC_XWR68XX
#define MMW_SOC_PIN_UART1_TX        SOC_XWR68XX_PINN5_PADBE
#define MMW_SOC_PIN_UART1_TX_FUNC   SOC_XWR68XX_PINN5_PADBE_MSS_UARTA_TX
#define MMW_SOC_PIN_UART1_RX        SOC_XWR68XX_PINN4_PADBD
#define MMW_SOC_PIN_UART1_RX_FUNC   SOC_XWR68XX_PINN4_PADBD_MSS_UARTA_RX
#define MMW_SOC_PIN_UART3_TX        SOC_XWR68XX_PINF14_PADAJ
#define MMW_SOC_PIN_UART3_TX_FUNC   SOC_XWR68XX_PINF14_PADAJ_MSS_UARTB_TX
#define MMW_SOC_PIN_GPIO2           SOC_XWR68XX_PINK13_PADAZ
#define MMW_SOC_PIN_GPIO2_FUNC      SOC_XWR68XX_PINK13_PADAZ_GPIO_2
#define MMW_SOC_PIN_GPIO2_OUT       SOC_XWR68XX_GPIO_2
#define MMW_MMWAVELINK_DEVICE_MAP   RL_AR_DEVICETYPE_16XX
#define MMW_MSS_FRAME_START_INT     SOC_XWR68XX_MSS_FRAME_START_INT
#endif
/* definition for xWR64xx device variant */
#ifdef SOC_XWR64XX
#define MMW_SOC_PIN_UART1_TX        SOC_XWR64XX_PINN5_PADBE
#define MMW_SOC_PIN_UART1_TX_FUNC   SOC_XWR64XX_PINN5_PADBE_MSS_UARTA_TX
#define MMW_SOC_PIN_UART1_RX        SOC_XWR64XX_PINN4_PADBD
#define MMW_SOC_PIN_UART1_RX_FUNC   SOC_XWR64XX_PINN4_PADBD_MSS_UARTA_RX
#define MMW_SOC_PIN_UART3_TX        SOC_XWR64XX_PINF14_PADAJ
#define MMW_SOC_PIN_UART3_TX_FUNC   SOC_XWR64XX_PINF14_PADAJ_MSS_UARTB_TX
#define MMW_SOC_PIN_GPIO2           SOC_XWR64XX_PINK13_PADAZ
#define MMW_SOC_PIN_GPIO2_FUNC      SOC_XWR64XX_PINK13_PADAZ_GPIO_2
#define MMW_SOC_PIN_GPIO2_OUT       SOC_XWR64XX_GPIO_2
#define MMW_MMWAVELINK_DEVICE_MAP   RL_AR_DEVICETYPE_16XX
#define MMW_MSS_FRAME_START_INT     SOC_XWR64XX_MSS_FRAME_START_INT
#endif

#define MMW_MSS_UNUSED_VAR(a)       {a=a;}

/**
 * @brief
 *  Millimeter Wave Demo Sensor State
 *
 * @details
 *  The enumeration is used to define the sensor states used in mmwDemo
 */
typedef enum MmwDemo_SensorState_e
{
    /*!  @brief Inital state after sensor is initialized.
     */
    MmwDemo_SensorState_INIT = 0,

    /*!  @brief Inital state after sensor is post RF init.
     */
    MmwDemo_SensorState_OPENED,

    /*!  @brief Indicates sensor is started */
    MmwDemo_SensorState_STARTED,

    /*!  @brief  State after sensor has completely stopped */
    MmwDemo_SensorState_STOPPED
}MmwDemo_SensorState;

/**
 * @brief Task handles storage structure
 */
typedef struct MmwDemo_TaskHandles_t
{
    /*! @brief   MMWAVE Control Task Handle */
    Task_Handle monReportTask;

    Task_Handle periodDiagTask;

    /*! @brief   Demo init task */
    Task_Handle initTask;
} MmwDemo_taskHandles;

typedef struct MmwRf_Cfg_t
{
    rlRfLdoBypassCfg_t          *rfLdoBypass;
    rlChanCfg_t                 *rfChannelCfg;
    rlAdcOutCfg_t               *adcOutCfg;
    rlLowPowerModeCfg_t         *lowPowrModeCfg;
    rlProfileCfg_t              *profileCfg;
    uint8_t                     profileCfgCnt;
    rlChirpCfg_t                *chirpCfg;
    uint16_t                    chirpCfgCnt;
    rlFrameCfg_t                *frameCfg;
    rlRfCalMonTimeUntConf_t     *rfCalMonTimeCfg;
    rlRunTimeCalibConf_t        *runTimeCalibCfg;

}MmwRf_Cfg;


/**
 * @brief
 *  Millimeter Wave Demo MCB
 *
 * @details
 *  The structure is used to hold all the relevant information for the
 *  Millimeter Wave demo.
 */
typedef struct MmwDemo_MSS_MCB_t
{
    /*! @brief      Configuration which is used to execute the demo */
    MmwDemo_Cfg               cfg;

    MmwRf_Cfg                 rfCfg;

    /*! * @brief    Handle to the SOC Module */
    SOC_Handle                  socHandle;

    /*!@brief   Handle to the peer Mailbox */
    Mbox_Handle                 peerMailbox;

    /*! @brief   Semaphore handle for the mailbox communication */
    SemaphoreP_Handle            mboxSemHandle;

    /*! @brief   DSS event handle */
    Event_Handle                eventHandle;

    /**
     * @brief   Flag which determines if the CLI Write should use the UART
     * in polled or blocking mode.
     */
    bool                        uartPolledMode;

    /*! @brief      UART Command Rx/Tx Handle */
    UART_Handle                 commandUartHandle;

    /*! @brief      Task handle storage */
    MmwDemo_taskHandles         taskHandles;

    /*! @brief   Rf frequency scale factor, = 2.7 for 60GHz device, = 3.6 for 76GHz device */
    double                      rfFreqScaleFactor;
} MmwDemo_MSS_MCB;

/**************************************************************************
 *************************** Extern Definitions ***************************
 **************************************************************************/
extern int32_t RFMon_reportHandler(uint16_t msgId, uint16_t asyncSB, uint8_t *payload);
extern void MmwaveLink_setLogFunc(void * func);

/* Functions to handle the actions need to move the sensor state */
extern int32_t MmwDemo_openSensor(bool isFirstTimeOpen);
extern int32_t MmwDemo_configSensor(void);
extern int32_t MmwDemo_startSensor(void);
extern void MmwDemo_stopSensor(void);

/* functions to manage the dynamic configuration */
extern uint8_t MmwDemo_isAllCfgInPendingState(void);
extern uint8_t MmwDemo_isAllCfgInNonPendingState(void);
extern void MmwDemo_resetStaticCfgPendingState(void);
extern void MmwDemo_CfgUpdate(void *srcPtr, uint32_t offset, uint32_t size, int8_t subFrameNum);

/* Debug Functions */
extern void _MmwDemo_debugAssert(int32_t expression, const char *file, int32_t line);
#define MmwDemo_debugAssert(expression) {                                      \
                                         _MmwDemo_debugAssert(expression,      \
                                                  __FILE__, __LINE__);         \
                                         DebugP_assert(expression);             \
                                        }

#ifdef __cplusplus
}
#endif

#endif /* DIAG_MON_MSS_H */

