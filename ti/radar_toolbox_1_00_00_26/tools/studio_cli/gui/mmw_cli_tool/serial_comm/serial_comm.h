/*
 * File name: serial_win32.h
 * Date: 2017/04/19 09:54
 * Author: Pavel Paces
 */
 
#include "windows.h"
 
#ifndef __SERIAL_WIN32_H__
#define __SERIAL_WIN32_H__
 
#ifdef __cplusplus
extern "C" {
#endif
 
// opens the serial line
HANDLE serialWin32_open(char* comPort);// LPCTSTR lpszDevice );
 
// closes the serial line
int serialWin32_close( HANDLE hSerial );
 
// sends a character
int serialWin32_putc( HANDLE hSerial, char c);
 
// receives the character 
// - returns 1 and content of c if sucessful
// - returns 0 in case of no data on the interface
int serialWin32_getc( HANDLE hSerial, char * c);
 
int serialWin32_read( HANDLE hSerial, char *rspStr, int length);
int serialWin32_write( HANDLE hSerial, char *cmdStr, int length);
#ifdef __cplusplus
}
#endif
 
#endif // __SERIAL_WIN32_H__