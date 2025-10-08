/* C runtime includes */
#include <stdio.h>
#include <stdarg.h>
#include <string.h>

/* lwIP core includes */
#include <ti/transport/lwip/lwip-stack/src/include/lwip/opt.h>

#include <ti/transport/lwip/lwip-stack/src/include/lwip/sys.h>
#include <ti/transport/lwip/lwip-stack/src/include/lwip/timeouts.h>
#include <ti/transport/lwip/lwip-stack/src/include/lwip/debug.h>
#include <ti/transport/lwip/lwip-stack/src/include/lwip/stats.h>
#include <ti/transport/lwip/lwip-stack/src/include/lwip/init.h>
#include <ti/transport/lwip/lwip-stack/src/include/lwip/tcpip.h>
#include <ti/transport/lwip/lwip-stack/src/include/lwip/netif.h>
#include <ti/transport/lwip/lwip-stack/src/include/lwip/api.h>

#include <ti/transport/lwip/lwip-stack/src/include/lwip/tcp.h>
#include <ti/transport/lwip/lwip-stack/src/include/lwip/dhcp.h>
#include <ti/transport/lwip/lwip-stack/src/include/lwip/autoip.h>

/* lwIP netif includes */
#include <ti/transport/lwip/lwip-stack/src/include/lwip/etharp.h>
#include <ti/transport/lwip/lwip-stack/src/include/netif/ethernet.h>

/* applications includes */
#include "tcpclient.h"

#include "default_netif.h"

/*
 * Using CPSW defined macros for this specific file
 */
#undef htons
#undef ntohs
#undef htonl
#undef ntohl

#include "tcpclient.h"
#include <ti/drv/enet/enet.h>
#include <ti/drv/enet/include/per/cpsw.h>
#if !(defined(SOC_AM273X) || defined(SOC_AWR294X))
#include <ti/drv/udma/udma.h>
#endif
#include <ti/drv/enet/examples/utils/include/enet_apputils.h>
#include <ti/drv/enet/examples/utils/include/enet_apputils_rtos.h>
#include <ti/drv/enet/examples/utils/include/enet_appmemutils.h>
#include <ti/drv/enet/examples/utils/include/enet_appboardutils.h>
#include <ti/drv/enet/examples/utils/include/enet_mcm.h>
#include <ti/drv/enet/examples/utils/include/enet_appsoc.h>
#include <ti/drv/enet/examples/utils/include/enet_apprm.h>
#if defined(SOC_AM273X)
#include <ti/drv/enet/examples/utils/include/enet_board_tpr12evm.h>
#elif defined(SOC_AWR294X)
#include <ti/drv/enet/examples/utils/include/enet_board_awr294xevm.h>
#endif

#include <ti/drv/enet/lwipif/inc/lwip2lwipif.h>
#include <ti/drv/enet/lwipif/inc/lwipif2enet_appif.h>

#include <ti/transport/lwip/lwip-stack/src/include/netif/ppp/ppp_opts.h>

/* include the port-dependent configuration */
// #include <ti/drv/enet/lwipif/inc/lwipcfg.h>
// #include <ti/drv/enet/lwipif/inc/lwipcfg.h>
#include "lwipcfg.h"

#ifndef LWIP_EXAMPLE_APP_ABORT
#define LWIP_EXAMPLE_APP_ABORT() 0
#endif

/** Define this to 1 to enable a port-specific ethernet interface as default interface. */
#ifndef USE_DEFAULT_ETH_NETIF
#define USE_DEFAULT_ETH_NETIF 1
#endif

/** Define this to 1 to enable a PPP interface. */
#ifndef USE_PPP
#define USE_PPP 0
#endif

/** Define this to 1 or 2 to support 1 or 2 SLIP interfaces. */
#ifndef USE_SLIPIF
#define USE_SLIPIF 0
#endif

/** Use an ethernet adapter? Default to enabled if port-specific ethernet netif or PPPoE are used. */
#ifndef USE_ETHERNET
#define USE_ETHERNET  (USE_DEFAULT_ETH_NETIF || PPPOE_SUPPORT)
#endif

/** Use an ethernet adapter for TCP/IP? By default only if port-specific ethernet netif is used. */
#ifndef USE_ETHERNET_TCPIP
#define USE_ETHERNET_TCPIP  (USE_DEFAULT_ETH_NETIF)
#endif

#if USE_SLIPIF
#include <netif/slipif.h>
#endif /* USE_SLIPIF */

#ifndef USE_DHCP
#define USE_DHCP    LWIP_DHCP
#endif
#ifndef USE_AUTOIP
#define USE_AUTOIP  LWIP_AUTOIP
#endif

#define APP_ENABLE_STATIC_CFG (1U)
#define ENETNIMUAPP_PACKET_POLL_PERIOD_US (1000U)


/* ========================================================================== */
/*                           Macros & Typedefs                                */
/* ========================================================================== */

#define CPSWLWIPIFAPP_PACKET_POLL_PERIOD_US (500U)
#define CPSWLWIPIFAPP_TSK_STACK_CPU_LOAD  (5U * 1024U)

#if !defined(BAREMETAL)
#define APP_TSK_STACK_MAIN              (4U * 1024U)
#endif

void enetTask(void * a0, void * a1);

