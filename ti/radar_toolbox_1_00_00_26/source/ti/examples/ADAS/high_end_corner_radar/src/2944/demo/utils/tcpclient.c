/*
 * Copyright (c) 2001-2003 Swedish Institute of Computer Science.
 * All rights reserved. 
 * 
 * Redistribution and use in source and binary forms, with or without modification, 
 * are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. The name of the author may not be used to endorse or promote products
 *    derived from this software without specific prior written permission. 
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED 
 * WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF 
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT 
 * SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, 
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT 
 * OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING 
 * IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY 
 * OF SUCH DAMAGE.
 *
 * This file is part of the lwIP TCP/IP stack.
 * 
 * Author: Adam Dunkels <adam@sics.se>
 *
 */
#include "tcpclient.h"

#include <ti/transport/lwip/lwip-stack/src/include/lwip/opt.h>
#include <ti/drv/enet/enet.h>

#include <string.h>

#if 1 //LWIP_NETCONN

#include <ti/transport/lwip/lwip-stack/src/include/lwip/sys.h>
#include <ti/transport/lwip/lwip-stack/src/include/lwip/api.h>
#ifdef SOC_AM273X
// #include <ti/demo/tpr12/mmw/mss/mmw_mss.h>
#elif defined (SOC_AWR294X)
#include "../mss/mmw_mss.h"
#endif

SemaphoreP_Handle objDataSemaphoreHandle;
extern MmwDemo_enetStreamObjData gEnetStreamObjData;
extern MmwDemo_MSS_MCB gMmwMssMCB;
/*-----------------------------------------------------------------------------------*/
static void 
tcpclient_thread(void *arg)
{
  struct netconn *conn;
  err_t err;
  LWIP_UNUSED_ARG(arg);

  /* Create a new connection identifier. */
#if LWIP_IPV6
  conn = netconn_new(NETCONN_TCP_IPV6);
  netconn_bind(conn, IP6_ADDR_ANY, 7);
#else /* LWIP_IPV6 */
  conn = netconn_new(NETCONN_TCP);
#endif /* LWIP_IPV6 */
  LWIP_ERROR("tcpclient: invalid conn", (conn != NULL), return;);

  SemaphoreP_pend(gMmwMssMCB.enetCfg.EnetCfgDoneSemHandle, SemaphoreP_WAIT_FOREVER);
  const ip_addr_t test_local_ip = gMmwMssMCB.enetCfg.remoteIp; 
  err = netconn_connect(conn, &test_local_ip ,7);
  if (err != ERR_OK){
    printf("netconn connect has failed !!\n");
  }

  if (err == ERR_OK) {
      while(1){
        /* Pending on the semaphore: Waiting for events to be received */
        SemaphoreP_pend (objDataSemaphoreHandle, SemaphoreP_WAIT_FOREVER);
        err = netconn_write(conn, &(gEnetStreamObjData.numObj), sizeof(uint32_t), NETCONN_COPY);
        err = netconn_write(conn, (gEnetStreamObjData.objData), sizeof(DPIF_PointCloudCartesian) * gEnetStreamObjData.numObj, NETCONN_COPY);
      
#if 0
          if (err != ERR_OK) {
            printf("tcpclient: netconn_write: error \"%s\"\n", lwip_strerr(err));
          }
#endif
      }
  }
}

/*-----------------------------------------------------------------------------------*/
void
tcpclient_init(void)
{
  SemaphoreP_Params semParams;

  /* Initialize timer semaphore params */
  SemaphoreP_Params_init(&semParams);
  semParams.mode = SemaphoreP_Mode_COUNTING;

  /* Create timer semaphore */
  objDataSemaphoreHandle = SemaphoreP_create(0, &semParams);
   
  sys_thread_new("tcpclient_thread", tcpclient_thread, NULL, DEFAULT_THREAD_STACKSIZE, DEFAULT_THREAD_PRIO);
}
/*-----------------------------------------------------------------------------------*/

#endif /* LWIP_NETCONN */
