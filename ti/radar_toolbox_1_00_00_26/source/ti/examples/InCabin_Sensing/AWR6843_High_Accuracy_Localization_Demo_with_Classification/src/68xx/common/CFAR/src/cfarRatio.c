/*!
 *  \file   cfarRatio.c
 *
 *  \brief   Source file for cfar angle bin weight ratios.
 *
 *  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
 *
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
 *
*/

#include "cfarRatio.h"

float cfarRatios[] = {1, 1, 1, 1, 1, 1.5, 1.8, 1.98, 2, 1.98, 1.8, 1.5, 1, 1, 1, 1, 1};

int32_t cfarRatio_matrixInit(float *ratioMatrix, uint8_t angleDim1, uint8_t angleDim2)
{
    uint32_t i, j;

    if (angleDim1 != angleDim2)
    {
        return -1;
    }

    if (angleDim1 != sizeof(cfarRatios)/sizeof(float))
    {
        return -2;
    }

    for (i = 0; i < angleDim1; i++)
    {
        ratioMatrix[i * angleDim1] = 1;
        for (j = 1; j < angleDim2; j++)
        {
            if (cfarRatios[i] == 1 || cfarRatios[j] == 1)
            {
                ratioMatrix[i * angleDim1 + j] = 1;
            }
            else
            {
                ratioMatrix[i * angleDim1 + j] = cfarRatios[i] * cfarRatios[j];
            }
        }
    }
    return 0;
}
