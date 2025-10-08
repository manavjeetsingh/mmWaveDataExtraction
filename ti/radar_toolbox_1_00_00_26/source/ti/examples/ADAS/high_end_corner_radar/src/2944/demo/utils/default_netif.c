#include <lwip/mem.h>
#include <lwip/tcpip.h>
#include <lwip2lwipif.h>

void init_default_netif(const ip4_addr_t *ipaddr, const ip4_addr_t *netmask, const ip4_addr_t *gw)
{
    /*TODO Make thread safe*/
    netif_default = (struct netif*)mem_malloc(sizeof(struct netif));
    netif_add(netif_default, ipaddr, netmask, gw, NULL, LWIPIF_LWIP_init, tcpip_input);
}

void default_netif_poll(void)
{
}

void default_netif_shutdown(void)
{
    netif_remove(netif_default);
}

