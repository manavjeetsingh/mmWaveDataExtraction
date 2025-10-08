;
; OSAL C674 Interrupt Support
;
; This implements the OSAL Interrupt Vector Table support for the
; C674
;
; Copyright (C) 2019 Texas Instruments Incorporated - http://www.ti.com/
; ALL RIGHTS RESERVED
;

;***************************************************************
;* Global Definitions:
;***************************************************************
    .global	_c_int00
    .global	OSAL_C674_Interrupt_vectors
    .global	OSAL_C674_Interrupt_NMIHandler
    .global	OSAL_C674_Interrupt_reserved
    .global	OSAL_C674_Interrupt_int4Handler
    .global	OSAL_C674_Interrupt_int5Handler
    .global	OSAL_C674_Interrupt_int6Handler
    .global	OSAL_C674_Interrupt_int7Handler
    .global	OSAL_C674_Interrupt_int8Handler
    .global	OSAL_C674_Interrupt_int9Handler
    .global	OSAL_C674_Interrupt_int10Handler
    .global	OSAL_C674_Interrupt_int11Handler
    .global	OSAL_C674_Interrupt_int12Handler
    .global	OSAL_C674_Interrupt_int13Handler
    .global	OSAL_C674_Interrupt_int14Handler
    .global	OSAL_C674_Interrupt_int15Handler

;***************************************************************
;* C674 Interrupt Vectors:
;***************************************************************
    .sect ".intvecs"
    .align 0x400
    .nocmp
    .def OSAL_C674_Interrupt_vectors

OSAL_C674_Interrupt_vectors:
OSAL_C674_Interrupt_reset:
    nop
    nop
    nop
    mvkl    _c_int00, b0
    mvkh    _c_int00, b0
    b       b0
    nop
    nop     4
;    b init_regs

OSAL_C674_Interrupt_nmi:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_NMIHandler, b0
    mvkh    OSAL_C674_Interrupt_NMIHandler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_reserved1:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_reserved, b0
    mvkh    OSAL_C674_Interrupt_reserved, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_reserved2:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_reserved, b0
    mvkh    OSAL_C674_Interrupt_reserved, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int4:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int4Handler, b0
    mvkh    OSAL_C674_Interrupt_int4Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int5:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int5Handler, b0
    mvkh    OSAL_C674_Interrupt_int5Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int6:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int6Handler, b0
    mvkh    OSAL_C674_Interrupt_int6Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int7:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int7Handler, b0
    mvkh    OSAL_C674_Interrupt_int7Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int8:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int8Handler, b0
    mvkh    OSAL_C674_Interrupt_int8Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int9:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int9Handler, b0
    mvkh    OSAL_C674_Interrupt_int9Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int10:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int10Handler, b0
    mvkh    OSAL_C674_Interrupt_int10Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int11:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int11Handler, b0
    mvkh    OSAL_C674_Interrupt_int11Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int12:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int12Handler, b0
    mvkh    OSAL_C674_Interrupt_int12Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int13:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int13Handler, b0
    mvkh    OSAL_C674_Interrupt_int13Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int14:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int14Handler, b0
    mvkh    OSAL_C674_Interrupt_int14Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4
OSAL_C674_Interrupt_int15:
    stw     b0, *b15--[1]
    nop
    nop
    mvkl    OSAL_C674_Interrupt_int15Handler, b0
    mvkh    OSAL_C674_Interrupt_int15Handler, b0
    b       b0
    ldw     *++b15[1], b0
    nop     4



init_regs:
        MVKL 0,B1
        MVKH 0,B1
        MVKL 0,A0
        MVKH 0,A0
        MV B1,B2
        MV B1,B3
    ||    MV B2,B4
        MV B1,B5
    ||    MV B2,B6
        MV B1,B7
    ||    MV B2,B8
        MV B1,B9
    ||    MV B2,B10
        MV B1,B11
    ||    MV B2,B12
        MV B1,B13
    ||    MV B2,B14
        MV B1,B15
    ||    MV B2,B16
        MV B1,B17
    ||    MV B2,B18
        MV B1,B19
    ||    MV B2,B20
        MV B1,B21
    ||    MV B2,B22
        MV B1,B23
    ||    MV B2,B24
        MV B1,B25
    ||    MV B2,B26
        MV B1,B27
    ||    MV B2,B28
        MV B1,B29
    ||    MV B2,B30
        MV B1,B31

        MV A0,A1
        MV A1,A2
        MV A1,A3
    ||    MV A2,A4
        MV A1,A5
    ||    MV A2,A6
        MV A1,A7
    ||    MV A2,A8
        MV A1,A9
    ||    MV A2,A10
        MV A1,A11
    ||    MV A2,A12
        MV A1,A13
    ||    MV A2,A14
        MV A1,A15
    ||    MV A2,A16
        MV A1,A17
    ||    MV A2,A18
        MV A1,A19
    ||    MV A2,A20
        MV A1,A21
    ||    MV A2,A22
        MV A1,A23
    ||    MV A2,A24
        MV A1,A25
    ||    MV A2,A26
        MV A1,A27
    ||    MV A2,A28
        MV A1,A29
    ||    MV A2,A30
        MV A1,A31

        NOP 5

        MVKL _c_int00,B0
        MVKH _c_int00,B0
        B B0

        NOP 5
        NOP 5
        NOP 5
