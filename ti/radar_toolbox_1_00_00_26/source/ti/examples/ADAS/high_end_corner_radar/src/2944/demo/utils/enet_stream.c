/*
 * Copyright (c) 2001,2002 Florian Schulze.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the authors nor the names of the contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * test.c - This file is part of lwIP test
 *
 */

#include "enet_stream.h"
#ifdef SOC_AM273X
// #include <ti/demo/tpr12/mmw/mss/mmw_mss.h>
#elif defined(SOC_AWR294X)
#include "../mss/mmw_mss.h"
#endif
#include <ti/utils/cli/cli.h>
/* ========================================================================== */
/*                         Structure Declarations                             */
/* ========================================================================== */

/* globales variables for netifs */
#if USE_ETHERNET
#if LWIP_DHCP
/* dhcp struct for the ethernet netif */
struct dhcp netif_dhcp;
#endif /* LWIP_DHCP */
#if LWIP_AUTOIP
/* autoip struct for the ethernet netif */
struct autoip netif_autoip;
#endif /* LWIP_AUTOIP */
#endif /* USE_ETHERNET */

typedef struct
{
    /* ENET instance type */
    Enet_Type enetType;

    /* ENET instance id */
    uint32_t instId;

    /* MAC port number */
    Enet_MacPort macPort;

    /* MII interface type */
    EnetMacPort_Interface mii;

    /* Id of the board where PHY is located */
    uint32_t boardId;

    bool useDfltFlow;
} EnetNimu_AppCfg;

typedef struct
{
    EnetMcm_CmdIf hMcmCmdIf[ENET_TYPE_NUM];

#if !(defined(SOC_AM273X) || defined(SOC_AWR294X))
    Udma_DrvHandle hUdmaDrv;
#endif
} EnetNimu_AppObj;

/* ========================================================================== */
/*                            Global Variables                                */
/* ========================================================================== */

static EnetNimu_AppCfg gEnetNimuAppCfg =
{
    .useDfltFlow = true,
};

static EnetNimu_AppObj gEnetNimuAppObj =
{
    .hMcmCmdIf =
    {
        [ENET_CPSW_2G] = {.hMboxCmd = NULL, .hMboxResponse = NULL},
        [ENET_CPSW_9G] = {.hMboxCmd = NULL, .hMboxResponse = NULL},
    },
};

extern MmwDemo_MSS_MCB gMmwMssMCB;

#if !defined(BAREMETAL)
static uint8_t  gAppTskStackMain[APP_TSK_STACK_MAIN] __attribute__((aligned(32)));
#endif
/* ========================================================================== */
/*                            Function Declaration                            */
/* ========================================================================== */

static void apps_init(void);

#if LWIP_NETIF_STATUS_CALLBACK
static void
status_callback(struct netif *state_netif)
{
  if (netif_is_up(state_netif)) {
#if LWIP_IPV4
    CLI_write("status_callback==UP, local interface IP is %s\n", ip4addr_ntoa(netif_ip4_addr(state_netif)));
    const ip4_addr_t *localIpTemp = netif_ip4_addr(state_netif);
    memcpy(&gMmwMssMCB.enetCfg.localIp, localIpTemp, sizeof(ip4_addr_t));
    gMmwMssMCB.enetCfg.status = 1;
    if(netif_ip4_addr(state_netif)->addr != 0){
        printf("Initializing apps\n");
        /* init apps */
        apps_init();
    }
#else
    printf("status_callback==UP\n");
#endif
  } else {
    printf("status_callback==DOWN\n");
  }
}
#endif /* LWIP_NETIF_STATUS_CALLBACK */

#if LWIP_NETIF_LINK_CALLBACK
static void
link_callback(struct netif *state_netif)
{
  if (netif_is_link_up(state_netif)) {
    printf("link_callback==UP\n");
  } else {
    printf("link_callback==DOWN\n");
  }
}
#endif /* LWIP_NETIF_LINK_CALLBACK */

/* This function initializes all network interfaces */
static void
test_netif_init(void)
{
#if LWIP_IPV4 && USE_ETHERNET
  ip4_addr_t ipaddr, netmask, gw;
#endif /* LWIP_IPV4 && USE_ETHERNET */
#if USE_DHCP || USE_AUTOIP
  err_t err;
#endif

#if USE_ETHERNET
#if LWIP_IPV4
  ip4_addr_set_zero(&gw);
  ip4_addr_set_zero(&ipaddr);
  ip4_addr_set_zero(&netmask);
#if USE_ETHERNET_TCPIP
#if USE_DHCP
  printf("Starting lwIP, local interface IP is dhcp-enabled\n");
#elif USE_AUTOIP
  printf("Starting lwIP, local interface IP is autoip-enabled\n");
#else /* USE_DHCP */
  LWIP_PORT_INIT_GW(&gw);
  LWIP_PORT_INIT_IPADDR(&ipaddr);
  LWIP_PORT_INIT_NETMASK(&netmask);
  printf("Starting lwIP, local interface IP is %s\n", ip4addr_ntoa(&ipaddr));
#endif /* USE_DHCP */
#endif /* USE_ETHERNET_TCPIP */
#else /* LWIP_IPV4 */
  printf("Starting lwIP, IPv4 disable\n");
#endif /* LWIP_IPV4 */

#if LWIP_IPV4
  init_default_netif(&ipaddr, &netmask, &gw);
#else
  init_default_netif();
#endif
#if LWIP_IPV6
  netif_create_ip6_linklocal_address(netif_default, 1);
#if LWIP_IPV6_AUTOCONFIG
  netif_default->ip6_autoconfig_enabled = 1;
#endif
  printf("ip6 linklocal address: %s\n", ip6addr_ntoa(netif_ip6_addr(netif_default, 0)));
#endif /* LWIP_IPV6 */
#if LWIP_NETIF_STATUS_CALLBACK
  netif_set_status_callback(netif_default, status_callback);
#endif /* LWIP_NETIF_STATUS_CALLBACK */
#if LWIP_NETIF_LINK_CALLBACK
  netif_set_link_callback(netif_default, link_callback);
#endif /* LWIP_NETIF_LINK_CALLBACK */

#if USE_ETHERNET_TCPIP
#if LWIP_AUTOIP
  autoip_set_struct(netif_default, &netif_autoip);
#endif /* LWIP_AUTOIP */
#if LWIP_DHCP
  dhcp_set_struct(netif_default, &netif_dhcp);
#endif /* LWIP_DHCP */
  netif_set_up(netif_default);
#if USE_DHCP
  err = dhcp_start(netif_default);
  LWIP_ASSERT("dhcp_start failed", err == ERR_OK);
#elif USE_AUTOIP
  err = autoip_start(netif_default);
  LWIP_ASSERT("autoip_start failed", err == ERR_OK);
#endif /* USE_DHCP */
#else /* USE_ETHERNET_TCPIP */
  /* Use ethernet for PPPoE only */
  netif.flags &= ~(NETIF_FLAG_ETHARP | NETIF_FLAG_IGMP); /* no ARP */
  netif.flags |= NETIF_FLAG_ETHERNET; /* but pure ethernet */
#endif /* USE_ETHERNET_TCPIP */
#endif /* USE_ETHERNET */
}

#if LWIP_DNS_APP && LWIP_DNS
static void
dns_found(const char *name, const ip_addr_t *addr, void *arg)
{
  LWIP_UNUSED_ARG(arg);
  printf("%s: %s\n", name, addr ? ipaddr_ntoa(addr) : "<not found>");
}

static void
dns_dorequest(void *arg)
{
  const char* dnsname = "3com.com";
  ip_addr_t dnsresp;
  LWIP_UNUSED_ARG(arg);

  if (dns_gethostbyname(dnsname, &dnsresp, dns_found, 0) == ERR_OK) {
    dns_found(dnsname, &dnsresp, 0);
  }
}
#endif /* LWIP_DNS_APP && LWIP_DNS */

/* This function initializes applications */
static void
apps_init(void)
{

#if LWIP_TCPECHO_APP
#if LWIP_NETCONN && defined(LWIP_TCPECHO_APP_NETCONN)
  tcpclient_init();
#else /* LWIP_NETCONN && defined(LWIP_TCPECHO_APP_NETCONN) */
  tcpecho_raw_init();
#endif
#endif /* LWIP_TCPECHO_APP && LWIP_NETCONN */
}

/* This function initializes this lwIP test. When NO_SYS=1, this is done in
 * the main_loop context (there is no other one), when NO_SYS=0, this is done
 * in the tcpip_thread context */
static void
test_init(void * arg)
{ /* remove compiler warning */
#if NO_SYS
  LWIP_UNUSED_ARG(arg);
#else /* NO_SYS */
  sys_sem_t *init_sem;
  LWIP_ASSERT("arg != NULL", arg != NULL);
  init_sem = (sys_sem_t*)arg;
#endif /* NO_SYS */

  /* init randomizer again (seed per thread) */
  srand((unsigned int)sys_now()/1000);

  /* init network interfaces */
  test_netif_init();

#if !NO_SYS
  sys_sem_signal(init_sem);
#endif /* !NO_SYS */
}

/* This is somewhat different to other ports: we have a main loop here:
 * a dedicated task that waits for packets to arrive. This would normally be
 * done from interrupt context with embedded hardware, but we don't get an
 * interrupt in windows for that :-) */
static void
main_loop(void * a0,
          void * a1)
{
#if !NO_SYS
  err_t err;
  sys_sem_t init_sem;
#endif /* NO_SYS */

  /* initialize lwIP stack, network interfaces and applications */
#if NO_SYS
  lwip_init();
  test_init(NULL);
#else /* NO_SYS */
  err = sys_sem_new(&init_sem, 0);
  LWIP_ASSERT("failed to create init_sem", err == ERR_OK);
  LWIP_UNUSED_ARG(err);
  tcpip_init(test_init, &init_sem);
  /* we have to wait for initialization to finish before
   * calling update_adapter()! */
  sys_sem_wait(&init_sem);
  sys_sem_free(&init_sem);
#endif /* NO_SYS */

#if (LWIP_SOCKET || LWIP_NETCONN) && LWIP_NETCONN_SEM_PER_THREAD
  netconn_thread_init();
#endif

}

void enetTask(void * a0, void * a1)
{
    TaskP_Params params;
    TaskP_Handle hMainLoopTask = NULL;

  printf("Debug: Enet Configuration       \n");

#if (defined(SOC_AM273X) || defined(SOC_AWR294X))
    gEnetNimuAppCfg.enetType = ENET_CPSW_2G;
    gEnetNimuAppCfg.instId   = 0U;
    gEnetNimuAppCfg.boardId  = ENETBOARD_CPB_ID;
    gEnetNimuAppCfg.macPort  = ENET_MAC_PORT_1; /* RGMII port */
    gEnetNimuAppCfg.mii.layerType    = ENET_MAC_LAYER_GMII;
    gEnetNimuAppCfg.mii.sublayerType = ENET_MAC_SUBLAYER_REDUCED;

    EnetAppUtils_enableClocks(gEnetNimuAppCfg.enetType, gEnetNimuAppCfg.instId);
#endif

  /* no stdio-buffering, please! */

  TaskP_Params_init(&params);
  params.name = (uint8_t *)"Main Loop";
  params.priority       = DEFAULT_THREAD_PRIO;
  params.stack          = gAppTskStackMain;
  params.stacksize      = sizeof(gAppTskStackMain);
  hMainLoopTask         = TaskP_create((void *)main_loop, &params);

  if (hMainLoopTask == NULL)
  {
	  printf("ERROR: hMainLoop Task is NULL\n");
  }

}

void CpswApp_initAleConfig(CpswAle_Cfg *aleConfig)
{
    int32_t status = ENET_SOK;

    aleConfig->modeFlags =
        (CPSW_ALE_CFG_MODULE_EN);
    aleConfig->agingCfg.autoAgingEn = TRUE;
    aleConfig->agingCfg.agingPeriodInMs = 1000;

    EnetAppUtils_assert(status == ENET_SOK);

    aleConfig->nwSecCfg.vid0ModeEn               = TRUE;
    aleConfig->vlanCfg.aleVlanAwareMode           = FALSE;
    aleConfig->vlanCfg.cpswVlanAwareMode          = FALSE;
    aleConfig->vlanCfg.unknownUnregMcastFloodMask = CPSW_ALE_ALL_PORTS_MASK;
    aleConfig->vlanCfg.unknownRegMcastFloodMask   = CPSW_ALE_ALL_PORTS_MASK;
    aleConfig->vlanCfg.unknownVlanMemberListMask  = CPSW_ALE_ALL_PORTS_MASK;
    aleConfig->policerGlobalCfg.policingEn    = true;
    aleConfig->policerGlobalCfg.yellowDropEn  = false;
    /*! Enables the ALE to drop the red colored packets. */
    aleConfig->policerGlobalCfg.redDropEn = false;
    /*! Policing match mode */
    aleConfig->policerGlobalCfg.policerNoMatchMode = CPSW_ALE_POLICER_NOMATCH_MODE_GREEN;
}

void EnetApp_initLinkArgs(EnetPer_PortLinkCfg *linkArgs,
                          Enet_MacPort macPort)
{
    EnetPhy_Cfg *phyCfg = &linkArgs->phyCfg;
    EnetMacPort_LinkCfg *linkCfg = &linkArgs->linkCfg;
    EnetMacPort_Interface *mii = &linkArgs->mii;
    EnetBoard_EthPort ethPort;
    const EnetBoard_PhyCfg *boardPhyCfg;
    int32_t status;

    /* Setup board for requested Ethernet port */
    ethPort.enetType = gEnetNimuAppCfg.enetType;
    ethPort.instId   = gEnetNimuAppCfg.instId;
    ethPort.macPort  = gEnetNimuAppCfg.macPort;
    ethPort.boardId  = gEnetNimuAppCfg.boardId;
    ethPort.mii      = gEnetNimuAppCfg.mii;

    status = EnetBoard_setupPorts(&ethPort, 1U);
    EnetAppUtils_assert(status == ENET_SOK);

    if (Enet_isCpswFamily(gEnetNimuAppCfg.enetType))
    {
        CpswMacPort_Cfg *macCfg = (CpswMacPort_Cfg *)linkArgs->macCfg;
        CpswMacPort_initCfg(macCfg);
        if (EnetMacPort_isSgmii(mii) || EnetMacPort_isQsgmii(mii))
        {
            macCfg->sgmiiMode = ENET_MAC_SGMIIMODE_SGMII_WITH_PHY;
        }
        else
        {
            macCfg->sgmiiMode = ENET_MAC_SGMIIMODE_INVALID;
        }
    }

    boardPhyCfg = EnetBoard_getPhyCfg(&ethPort);
    if (boardPhyCfg != NULL)
    {
        EnetPhy_initCfg(phyCfg);
        phyCfg->phyAddr     = boardPhyCfg->phyAddr;
        phyCfg->isStrapped  = boardPhyCfg->isStrapped;
        phyCfg->loopbackEn  = false;
        phyCfg->skipExtendedCfg = boardPhyCfg->skipExtendedCfg;
        phyCfg->extendedCfgSize = boardPhyCfg->extendedCfgSize;
        memcpy(phyCfg->extendedCfg, boardPhyCfg->extendedCfg, phyCfg->extendedCfgSize);
    }
    else
    {
        printf("No PHY configuration found for MAC port %u\n",
                           ENET_MACPORT_ID(ethPort.macPort));
        EnetAppUtils_assert(false);
    }

    mii->layerType     = ethPort.mii.layerType;
    mii->sublayerType  = ethPort.mii.sublayerType;
    mii->variantType   = ENET_MAC_VARIANT_FORCED;
    linkCfg->speed     = ENET_SPEED_AUTO;
    linkCfg->duplexity = ENET_DUPLEX_AUTO;

    if (Enet_isCpswFamily(gEnetNimuAppCfg.enetType))
    {
        CpswMacPort_Cfg *macCfg = (CpswMacPort_Cfg *)linkArgs->macCfg;

        if (EnetMacPort_isSgmii(mii) || EnetMacPort_isQsgmii(mii))
        {
            macCfg->sgmiiMode = ENET_MAC_SGMIIMODE_SGMII_WITH_PHY;
        }
        else
        {
            macCfg->sgmiiMode = ENET_MAC_SGMIIMODE_INVALID;
        }
    }
}

static void EnetNimuApp_portLinkStatusChangeCb(Enet_MacPort macPort,
                                               bool isLinkUp,
                                               void *appArg)
{
    printf("MAC Port %u: link %s\n",
                       ENET_MACPORT_ID(macPort), isLinkUp ? "up" : "down");
}

static void EnetNimuApp_mdioLinkStatusChange(Cpsw_MdioLinkStateChangeInfo *info,
                                             void *appArg)
{
    static uint32_t linkUpCount = 0;
    if ((info->linkChanged) && (info->isLinked))
    {
        linkUpCount++;
    }
}

static void EnetNimuApp_initEnetLinkCbPrms(Cpsw_Cfg *cpswCfg)
{

    cpswCfg->mdioLinkStateChangeCb     = EnetNimuApp_mdioLinkStatusChange;
    cpswCfg->mdioLinkStateChangeCbArg  = &gEnetNimuAppObj;

    cpswCfg->portLinkStatusChangeCb    = &EnetNimuApp_portLinkStatusChangeCb;
    cpswCfg->portLinkStatusChangeCbArg = &gEnetNimuAppObj;
}

void EnetApp_initAleConfig(CpswAle_Cfg *aleCfg)
{
    int32_t status = ENET_SOK;

    aleCfg->modeFlags = CPSW_ALE_CFG_MODULE_EN;
    aleCfg->agingCfg.autoAgingEn = true;
    aleCfg->agingCfg.agingPeriodInMs = 1000;
    EnetAppUtils_assert(status == ENET_SOK);

    aleCfg->nwSecCfg.vid0ModeEn                = true;
    aleCfg->vlanCfg.aleVlanAwareMode           = FALSE;
    aleCfg->vlanCfg.cpswVlanAwareMode          = FALSE;
    aleCfg->vlanCfg.unknownUnregMcastFloodMask = CPSW_ALE_ALL_PORTS_MASK;
    aleCfg->vlanCfg.unknownRegMcastFloodMask   = CPSW_ALE_ALL_PORTS_MASK;
    aleCfg->vlanCfg.unknownVlanMemberListMask  = CPSW_ALE_ALL_PORTS_MASK;
    aleCfg->policerGlobalCfg.policingEn        = true;
    aleCfg->policerGlobalCfg.yellowDropEn      = false;
    /* Enables the ALE to drop the red colored packets. */
    aleCfg->policerGlobalCfg.redDropEn         = false;
    /* Policing match mode */
    aleCfg->policerGlobalCfg.policerNoMatchMode = CPSW_ALE_POLICER_NOMATCH_MODE_GREEN;
}

static int32_t EnetNimuApp_init(Enet_Type enetType)
{
    int32_t status = ENET_SOK;
    EnetMcm_InitConfig enetMcmCfg;
    Cpsw_Cfg cpswCfg;
    EnetRm_ResCfg *resCfg;
    Enet_MacPort macPortList[] = {gEnetNimuAppCfg.macPort};
    uint8_t numMacPorts        = (sizeof(macPortList) / sizeof(macPortList[0U]));
    EnetAppUtils_assert(numMacPorts <=
                        Enet_getMacPortMax(gEnetNimuAppCfg.enetType, gEnetNimuAppCfg.instId));

#if (defined(SOC_AM273X) || defined(SOC_AWR294X))
    EnetCpdma_Cfg dmaCfg;
    dmaCfg.rxInterruptPerMSec = 2;
#else
    EnetUdma_Cfg dmaCfg;
    /* Open UDMA */
    gEnetNimuAppObj.hUdmaDrv = EnetAppUtils_udmaOpen(gEnetNimuAppCfg.enetType, NULL);
    EnetAppUtils_assert(NULL != gEnetNimuAppObj.hUdmaDrv);
    dmaCfg.rxChInitPrms.dmaPriority = UDMA_DEFAULT_RX_CH_DMA_PRIORITY;
    dmaCfg.hUdmaDrv = gEnetNimuAppObj.hUdmaDrv;
#endif

    /* Set configuration parameters */
    if (Enet_isCpswFamily(enetType))
    {
    	cpswCfg.dmaCfg = (void *)&dmaCfg;

        Enet_initCfg(gEnetNimuAppCfg.enetType, gEnetNimuAppCfg.instId, &cpswCfg, sizeof(cpswCfg));
        cpswCfg.vlanCfg.vlanAware          = false;
        cpswCfg.hostPortCfg.removeCrc      = true;
        cpswCfg.hostPortCfg.padShortPacket = true;
        cpswCfg.hostPortCfg.passCrcErrors  = true;
        EnetNimuApp_initEnetLinkCbPrms(&cpswCfg);
        resCfg = &cpswCfg.resCfg;
        EnetApp_initAleConfig(&cpswCfg.aleCfg);

        enetMcmCfg.perCfg = &cpswCfg;
    }

    EnetAppUtils_assert(NULL != enetMcmCfg.perCfg);
    EnetAppUtils_initResourceConfig(gEnetNimuAppCfg.enetType, EnetSoc_getCoreId(), resCfg);

    enetMcmCfg.enetType           = gEnetNimuAppCfg.enetType;
    enetMcmCfg.instId             = gEnetNimuAppCfg.instId;
    enetMcmCfg.setPortLinkCfg     = EnetApp_initLinkArgs;
    enetMcmCfg.numMacPorts        = numMacPorts;
    enetMcmCfg.periodicTaskPeriod = ENETPHY_FSM_TICK_PERIOD_MS; /* msecs */
    enetMcmCfg.print              = EnetAppUtils_print;

    memcpy(&enetMcmCfg.macPortList[0U], &macPortList[0U], sizeof(macPortList));
    status = EnetMcm_init(&enetMcmCfg);

    return status;
}

static bool EnetApp_isPortLinked(Enet_Handle hEnet)
{
    uint32_t coreId = EnetSoc_getCoreId();

    return EnetAppUtils_isPortLinkUp(hEnet, coreId, gEnetNimuAppCfg.macPort);
}

void LwipifEnetAppCb_getHandle(LwipifEnetAppIf_GetHandleInArgs *inArgs,
                             LwipifEnetAppIf_GetHandleOutArgs *outArgs)
{
    int32_t status;
    EnetMcm_HandleInfo handleInfo;
    EnetPer_AttachCoreOutArgs attachInfo;
    EnetMcm_CmdIf *pMcmCmdIf = &gEnetNimuAppObj.hMcmCmdIf[gEnetNimuAppCfg.enetType];

#if defined (ENET_SOC_HOSTPORT_DMA_TYPE_UDMA)
    EnetUdma_OpenRxFlowPrms enetRxFlowCfg;
    EnetUdma_OpenTxChPrms   enetTxChCfg;
    bool useRingMon          = false;
    bool useDfltFlow      = gEnetNimuAppCfg.useDfltFlow;
#elif defined (ENET_SOC_HOSTPORT_DMA_TYPE_CPDMA)
    EnetCpdma_OpenRxChPrms enetRxFlowCfg;
    EnetCpdma_OpenTxChPrms enetTxChCfg;
#endif
    uint32_t coreId          = EnetSoc_getCoreId();

    if (pMcmCmdIf->hMboxCmd == NULL)
    {
        status = EnetNimuApp_init(gEnetNimuAppCfg.enetType);

        if (status != ENET_SOK)
        {
            printf("Failed to open ENET: %d\n", status);
        }
        EnetAppUtils_assert(status == ENET_SOK);
        EnetMcm_getCmdIf(gEnetNimuAppCfg.enetType, pMcmCmdIf);
    }

    EnetAppUtils_assert(pMcmCmdIf->hMboxCmd != NULL);
    EnetAppUtils_assert(pMcmCmdIf->hMboxResponse != NULL);
    EnetMcm_acquireHandleInfo(pMcmCmdIf, &handleInfo);
    EnetMcm_coreAttach(pMcmCmdIf, coreId, &attachInfo);

    /* Confirm HW checksum offload is enabled as NIMU enables it by default */
    if (Enet_isCpswFamily(gEnetNimuAppCfg.enetType))
    {
        Enet_IoctlPrms prms;
        bool csumOffloadFlg;
        ENET_IOCTL_SET_OUT_ARGS(&prms, &csumOffloadFlg);
        status = Enet_ioctl(handleInfo.hEnet,
                            coreId,
                            ENET_HOSTPORT_IS_CSUM_OFFLOAD_ENABLED,
                            &prms);
        if (status != ENET_SOK)
        {
            printf("() Failed to get checksum offload info: %d\n", status);
        }

        EnetAppUtils_assert(true == csumOffloadFlg);
    }

    /* Open TX channel */
    EnetDma_initTxChParams(&enetTxChCfg);

#if defined(ENET_SOC_HOSTPORT_DMA_TYPE_UDMA)
    enetTxChCfg.hUdmaDrv  = handleInfo.hUdmaDrv;
    enetTxChCfg.useProxy  = true;
#endif
    enetTxChCfg.numTxPkts = inArgs->txCfg.numPackets;
    enetTxChCfg.cbArg     = inArgs->txCfg.cbArg;
    enetTxChCfg.notifyCb  = inArgs->txCfg.notifyCb;
    EnetAppUtils_setCommonTxChPrms(&enetTxChCfg);

    EnetAppUtils_openTxCh(handleInfo.hEnet,
                          attachInfo.coreKey,
                          coreId,
                          &outArgs->txInfo.txChNum,
                          &outArgs->txInfo.hTxChannel,
                          &enetTxChCfg);
    /* Open RX Flow */
    EnetDma_initRxChParams(&enetRxFlowCfg);
    enetRxFlowCfg.notifyCb  = inArgs->rxCfg.notifyCb;
    enetRxFlowCfg.numRxPkts = inArgs->rxCfg.numPackets;
    enetRxFlowCfg.cbArg     = inArgs->rxCfg.cbArg;
#if defined (ENET_SOC_HOSTPORT_DMA_TYPE_UDMA)
    enetRxFlowCfg.hUdmaDrv  = handleInfo.hUdmaDrv;
    enetRxFlowCfg.useProxy  = true;

#if (UDMA_SOC_CFG_UDMAP_PRESENT == 1)
    /* Use ring monitor for the CQ ring of RX flow */
    EnetUdma_UdmaRingPrms *pFqRingPrms = &enetRxFlowCfg.udmaChPrms.fqRingPrms;
    pFqRingPrms->useRingMon = true;
    pFqRingPrms->ringMonCfg.mode = TISCI_MSG_VALUE_RM_MON_MODE_THRESHOLD;
    /* Ring mon low threshold */

#if defined _DEBUG_
    /* In debug mode as CPU is processing lesser packets per event, keep threshold more */
    pFqRingPrms->ringMonCfg.data0 = (inArgs->rxCfg.numPackets - 10U);
#else
    pFqRingPrms->ringMonCfg.data0 = (inArgs->rxCfg.numPackets - 20U);
#endif
    /* Ring mon high threshold - to get only low  threshold event, setting high threshold as more than ring depth*/
    pFqRingPrms->ringMonCfg.data1 = inArgs->rxCfg.numPackets;
#endif

    EnetAppUtils_setCommonRxFlowPrms(&enetRxFlowCfg);

    EnetAppUtils_openRxFlow(gEnetNimuAppCfg.enetType,
                            handleInfo.hEnet,
                            attachInfo.coreKey,
                            coreId,
                            useDfltFlow,
                            &outArgs->rxInfo.rxFlowStartIdx,
                            &outArgs->rxInfo.rxFlowIdx,
                            &outArgs->rxInfo.macAddr[0U],
                            &outArgs->rxInfo.hRxFlow,
                            &enetRxFlowCfg);

#elif defined (ENET_SOC_HOSTPORT_DMA_TYPE_CPDMA)
    EnetAppUtils_setCommonRxChPrms(&enetRxFlowCfg);
    EnetAppUtils_openRxCh(handleInfo.hEnet,
    		attachInfo.coreKey,
			coreId,
			&outArgs->rxInfo.rxFlowIdx,
			&outArgs->rxInfo.hRxFlow,
			&enetRxFlowCfg);
#endif

    outArgs->coreId        = coreId;
    outArgs->hEnet         = handleInfo.hEnet;
    outArgs->hostPortRxMtu = attachInfo.rxMtu;
    ENET_UTILS_ARRAY_COPY(outArgs->txMtu, attachInfo.txMtu);
    outArgs->coreKey       = attachInfo.coreKey;
#if (defined(SOC_AM273X) || defined(SOC_AWR294X))
    outArgs->isRingMonUsed = false;
#else
    outArgs->hUdmaDrv       = handleInfo.hUdmaDrv;
    outArgs->isRingMonUsed = useRingMon;
#endif
	outArgs->print          = &EnetAppUtils_print;
    outArgs->isPortLinkedFxn = &EnetApp_isPortLinked;

    outArgs->timerPeriodUs   = ENETNIMUAPP_PACKET_POLL_PERIOD_US;

    /* Let NIMU use optimized processing where TX packets are relinquished in next
     * TX submit call */
    outArgs->disableTxEvent = true;
    printf("Host MAC address: ");
#if (defined(SOC_AM273X) || defined(SOC_AWR294X))
    //FIXME - TPR12 openRxCh doesn't allocate MAC addresses which seems correct behaviour
    status = EnetAppUtils_allocMac(handleInfo.hEnet,
    		outArgs->coreKey,
			outArgs->coreId,
			&outArgs->rxInfo.macAddr[0U]);
#endif
    uint8_t * macAddr = &outArgs->rxInfo.macAddr[0U];
    printf("%02x:%02x:%02x:%02x:%02x:%02x\n",
                       macAddr[0] & 0xFF,
                       macAddr[1] & 0xFF,
                       macAddr[2] & 0xFF,
                       macAddr[3] & 0xFF,
                       macAddr[4] & 0xFF,
                       macAddr[5] & 0xFF);

}

void LwipifEnetAppCb_releaseHandle(LwipifEnetAppIf_ReleaseHandleInfo *releaseInfo)
{
    EnetDma_PktQ fqPktInfoQ;
    EnetDma_PktQ cqPktInfoQ;

#if !(defined(SOC_AM273X) || defined(SOC_AWR294X))
    bool useDfltFlow = gEnetNimuAppCfg.useDfltFlow;
#endif
    EnetMcm_CmdIf *pMcmCmdIf = &gEnetNimuAppObj.hMcmCmdIf[gEnetNimuAppCfg.enetType];

    EnetAppUtils_assert(pMcmCmdIf->hMboxCmd != NULL);
    EnetAppUtils_assert(pMcmCmdIf->hMboxResponse != NULL);

    /* Close TX channel */
    {
        EnetQueue_initQ(&fqPktInfoQ);
        EnetQueue_initQ(&cqPktInfoQ);
        EnetAppUtils_closeTxCh(releaseInfo->hEnet,
                               releaseInfo->coreKey,
                               releaseInfo->coreId,
                               &fqPktInfoQ,
                               &cqPktInfoQ,
                               releaseInfo->txInfo.hTxChannel,
                               releaseInfo->txInfo.txChNum);
        releaseInfo->txFreePktCb(releaseInfo->freePktCbArg, &fqPktInfoQ, &cqPktInfoQ);
    }

    {
        /* Close RX Flow */
        EnetQueue_initQ(&fqPktInfoQ);
        EnetQueue_initQ(&cqPktInfoQ);
#if !(defined(SOC_AM273X) || defined(SOC_AWR294X))
        EnetAppUtils_closeRxFlow(gEnetNimuAppCfg.enetType,
                                 releaseInfo->hEnet,
                                 releaseInfo->coreKey,
                                 releaseInfo->coreId,
                                 useDfltFlow,
                                 &fqPktInfoQ,
                                 &cqPktInfoQ,
                                 releaseInfo->rxInfo.rxFlowStartIdx,
                                 releaseInfo->rxInfo.rxFlowIdx,
                                 releaseInfo->rxInfo.macAddr,
                                 releaseInfo->rxInfo.hRxFlow);
#endif
        releaseInfo->rxFreePktCb(releaseInfo->freePktCbArg, &fqPktInfoQ, &cqPktInfoQ);
    }

    EnetMcm_coreDetach(pMcmCmdIf, releaseInfo->coreId, releaseInfo->coreKey);
    EnetMcm_releaseHandleInfo(pMcmCmdIf);
}


