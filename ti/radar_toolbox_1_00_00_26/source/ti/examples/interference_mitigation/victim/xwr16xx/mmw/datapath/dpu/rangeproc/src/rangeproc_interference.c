 /* 
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
/**
 *   @file  rangeproc_interference.c
 *
 *   @brief
 *      Implements interference related functions
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "../include/rangeproc_interference.h"

#define N_HYST 15


/* inverseLenArr and rightShiftArr are used to calculate the inverse of a number.
 * 1/num = inverseLenArr[num-2]/(2^rightShiftArr[num-2])
 * example: 1/2 = inverseLenArr[0]/(2^rightShiftArr[0]) = 128/2^8= 1/2
 */ 

static int16_t inverseLenArr[] = {  128, 171, 128, 205, 171,
									146, 128, 228, 205, 186,
									171, 158, 146, 137, 128, 
									241, 228, 216, 205, 195,
									186, 178, 171, 164, 158,
									152, 146, 141, 137, 132,    
									128, 248};

static int16_t rightShiftArr[] = {	8, 9, 9, 10, 10, 10, 10, 11, 11,
									11, 11, 11, 11, 11, 11, 12, 12, 12, 
									12, 12, 12, 12, 12, 12, 12, 12, 12, 
									12, 12, 12, 12, 13};	


/**
*  @b Description
*  @n
*    Create linear interpolation coefficients from IIB array 
*    For the i'th region to repair, lastGoodSampleLoc[i] stores the location of 
*    the last good ADC sample before the i'th region of interference and 
*    newestGoodSampleLoc[i] stores the location of the first good sample after
*    the i'th region of interference. This is calculated on the basis of interfDetectArr
*
*    Note: There shouldn't be more than a few regions of interference in realistic scenarios.
*
*  @retval
*      Number of regions to repair
*/
uint32_t  intfMitgCreateLinearInterpCoeffsFromIIB(uint8_t interfDetectArr[restrict],  uint32_t len,
	int16_t lastGoodSampleLoc[restrict], int16_t newestGoodSampleLoc[restrict])
{
	int32_t  sampleIdx;
	int32_t numRegionsToRepair = 0;
	if (interfDetectArr[0] == 1) 
	{
		//Interference detected at 0th index. last good sample before 0th index not available
		lastGoodSampleLoc[numRegionsToRepair] = -1;
	}
	
	
	for (sampleIdx = 1; sampleIdx < len - 1; sampleIdx++)
	{
		if ( (interfDetectArr[sampleIdx] == 1) && 
			 (interfDetectArr[sampleIdx - 1] == 0) )
		{
			lastGoodSampleLoc[numRegionsToRepair] = sampleIdx-1;
		}
		
		if ( (interfDetectArr[sampleIdx] == 1) && 
			 (interfDetectArr[sampleIdx + 1] == 0) )
		{
			newestGoodSampleLoc[numRegionsToRepair] = sampleIdx+1;
			numRegionsToRepair++;
		}
	}
	
	if (interfDetectArr[len-1] == 1) 
	{
		//Interference detected at (len-1)th index. newest good sample after (len-1)th index not available
		newestGoodSampleLoc[numRegionsToRepair] = -1;
		numRegionsToRepair++;
	}
	
	return numRegionsToRepair;
}

					  
/**
*  @b Description
*  @n
*    Perform linear interpolation for each of numRegionsToRepair.
*    This is done on the basis of the last good sample before a region of interference
*    and the newest good sample after the region of interference.
*
*  @retval
*      Not Applicable.
*/
uint32_t  intfMitgPerformLinearInterp(cmplx16ImRe_t inp[restrict], uint32_t len,
	int16_t lastGoodSampleLoc[restrict], int16_t newestGoodSampleLoc[restrict], int32_t numRegionsToRepair)
{
	int32_t  sampleIdx, regionToRepairIdx;
	int16_t regionLen, linear_interp_imag, linear_interp_real;
	int32_t slope_imag, slope_real, sum_val_imag, sum_val_real;

	//Loop over each region to be repaired
	for (regionToRepairIdx = 0; regionToRepairIdx < numRegionsToRepair; regionToRepairIdx++)
	{
		
		if ((lastGoodSampleLoc[regionToRepairIdx] == -1) &&
			(newestGoodSampleLoc[regionToRepairIdx] != -1))
		{
			//last good sample isn't available for this region. The region is filled out with
			//newest good sample
			for (sampleIdx = 0; sampleIdx < newestGoodSampleLoc[regionToRepairIdx]; sampleIdx++)
			{
				inp[sampleIdx] = inp[newestGoodSampleLoc[regionToRepairIdx]];
			}

		}
		else if ((lastGoodSampleLoc[regionToRepairIdx] != -1) &&
			(newestGoodSampleLoc[regionToRepairIdx] == -1))
		{
			//newest good sample isn't available for this region. The region is filled out with
			//last good sample
			for (sampleIdx = lastGoodSampleLoc[regionToRepairIdx] + 1; sampleIdx < len; sampleIdx++)
			{
				inp[sampleIdx] = inp[lastGoodSampleLoc[regionToRepairIdx]];
			}
		}
		else if ((lastGoodSampleLoc[regionToRepairIdx] != -1) &&
			(newestGoodSampleLoc[regionToRepairIdx] != -1))
		{
			//Both last and newest good samples available for the region.

			//Calculate the slope for line used in linear interpolation
			if ((newestGoodSampleLoc[regionToRepairIdx] - lastGoodSampleLoc[regionToRepairIdx]) < 32)
			{
				regionLen = newestGoodSampleLoc[regionToRepairIdx] - lastGoodSampleLoc[regionToRepairIdx];

				slope_real = (int32_t)(inp[newestGoodSampleLoc[regionToRepairIdx]].real -
					inp[lastGoodSampleLoc[regionToRepairIdx]].real);
				slope_real *= inverseLenArr[regionLen - 1];

				slope_imag = (int32_t)(inp[newestGoodSampleLoc[regionToRepairIdx]].imag -
					inp[lastGoodSampleLoc[regionToRepairIdx]].imag);

				slope_imag *= inverseLenArr[regionLen - 1];


				sum_val_real = 0;				sum_val_imag = 0;
				
				//Overwrite corrupted samples with interpolated ones
				for (sampleIdx = lastGoodSampleLoc[regionToRepairIdx] + 1; sampleIdx < newestGoodSampleLoc[regionToRepairIdx]; sampleIdx++)
				{
					sum_val_real += slope_real;
					sum_val_imag += slope_imag;

					linear_interp_real = sum_val_real >> (rightShiftArr[regionLen - 1]);
					linear_interp_imag = sum_val_imag >> (rightShiftArr[regionLen - 1]);

					inp[sampleIdx].real = inp[lastGoodSampleLoc[regionToRepairIdx]].real + (int16_t)linear_interp_real;
					inp[sampleIdx].imag = inp[lastGoodSampleLoc[regionToRepairIdx]].imag + (int16_t)linear_interp_imag;
				}
			}
		}

	}


	return 1;
}


/**
*  @b Description
*  @n
*    computes the thresholds threshAbs and threshAbsDiff based on the following formulae:
*    threshAbs = \frac{\sum_{1}^{len}\left | x_{i} \right | \times threshFacAbs}
*					{len \times 2^{threshFacBitwidth}}
*    and
*    threshAbsDiff = \frac{\sum_{2}^{len}\left | x_{i} - x_{i-1}\right | \times threshFacAbsDiff}
*					{len \times 2^{threshFacBitwidth}} 
*    x_i is the i’th ADC sample in a chirp with ‘len’ ADC samples. 
*    Only one RX channel is used for threshold computation. A sample is a 16-bit complex number.
*    threshAbs and threshAbsDiff are 16-bit numbers, and are both capped at a maximum of 0x7FFF.
*    thresholdAbsFac, thresholdAbsDiffFac and thresholdFacBitwidth are the input threshold factors
*
*  @retval
*      Not Applicable.
*/
void intfMitgThresholdComputation(cmplx16ImRe_t inp[restrict], uint32_t len,
	int16_t * pThreshAbs, int16_t * pThreshAbsDiff,
	uint16_t thresholdAbsFac, uint16_t thresholdAbsDiffFac, uint16_t thresholdFacBitwidth)
{
	uint32_t idx;
	int16_t realDiffPrev, imagDiffPrev, absValue;


	int32_t sumAbsActual = 0;
	int32_t sumAbsDiffActual = 0;

	cmplx16ImRe_t xl_inp, xl_tmp;

	realDiffPrev = inp[0].real;
	imagDiffPrev = inp[0].imag;

	_nassert(((uint32_t)inp % 8U) == 0);
	_nassert(((uint32_t)len) > 0);
	for (idx = 0; idx < len; idx++)
	{
		xl_inp = inp[idx];

		absValue = jplApprox(xl_inp);
		sumAbsActual += absValue;

		/* Diff. */
		xl_tmp.real = xl_inp.real - realDiffPrev;
		xl_tmp.imag = xl_inp.imag - imagDiffPrev;
		realDiffPrev = xl_inp.real;
		imagDiffPrev = xl_inp.imag;

		absValue = jplApprox(xl_tmp);

		sumAbsDiffActual += absValue;
	}

	{
		/* Threshold computation. */
		int32_t threshAbs = ((sumAbsActual * ((int32_t)thresholdAbsFac)) >> ((int32_t)thresholdFacBitwidth));
		int32_t threshAbsDiff = ((sumAbsDiffActual * ((int32_t)thresholdAbsDiffFac)) >> ((int32_t)thresholdFacBitwidth));
		threshAbs /= len;
		threshAbsDiff /= len;

		if (threshAbs > 0x7FFF)
		{
			threshAbs = 0x7FFF;
		}

		if (threshAbsDiff > 0x7FFF)
		{
			threshAbsDiff = 0x7FFF;
		}

		*pThreshAbs = (int16_t)(threshAbs);
		*pThreshAbsDiff = (int16_t)(threshAbsDiff);
	}
}


/**
*  @b Description
*  @n
*    creates an array interfDetectArr of interference indicator bits.
*    if corruption due to interference is detected in the i'th location,
*    interfDetectArr[i-N_HYST..i+N_HYST] is set to 1.
*    A sample inp[i] is detected to be corrupted by interference if:
*    |inp[i]| > threshAbs or
*    |inp[i]-inp[i-1]| > threshAbsDiff
*
*  @retval
*      Not Applicable.
*/
uint32_t intfMitgCreateIIBArr(cmplx16ImRe_t inp[restrict], uint8_t interfDetectArr[restrict], uint32_t len,
	int16_t threshAbs, int16_t threshAbsDiff)
{

	int32_t idx_inner;
	int32_t idx_outer;
	int16_t absValue, absDiffValue;
	// Array initialization.
	memset((void *)interfDetectArr, 0, (sizeof (uint8_t) * len));
	
	cmplx16ImRe_t xl_inp, DiffPrev;

	DiffPrev.real = inp[0].real;
	DiffPrev.imag = inp[0].imag;

	_nassert(((uint32_t)interfDetectArr % 8U) == 0);
	_nassert(((uint32_t)inp % 8U) == 0);
	for (idx_outer = 0; idx_outer < len; idx_outer += 1)
	{

		xl_inp = (inp[idx_outer]);
		absValue = jplApprox(xl_inp);

		xl_inp.real -= DiffPrev.real;
		xl_inp.imag -= DiffPrev.imag;
		absDiffValue = jplApprox(xl_inp);

		DiffPrev = inp[idx_outer];

		if ((absValue > threshAbs) ||
			(absDiffValue > threshAbsDiff))
		{
			#pragma UNROLL(2*N_HYST + 1)
			for (idx_inner = -N_HYST; idx_inner <= N_HYST; idx_inner ++)
			{
				if (((idx_outer + idx_inner) >= 0) && 
					 ((idx_outer + idx_inner) < len))
				{
					interfDetectArr[idx_outer + idx_inner] = 1;					
				}
			}
			idx_outer+= N_HYST;

		}
		
	}


	return 1;
}

/**
*  @b Description
*  @n
*    Applies interfDetectArr to the input.
*    It does so by zeroing out all samples inp[i] affected by interference, indicated by
*    interfDetectArr[i] being set to 1.
*
*  @retval
*      Not Applicable.
*/	
uint32_t intfMitgApplyIIBArr(cmplx16ImRe_t inp[restrict], uint8_t interfDetectArr[restrict], uint32_t len)
{
	
	int32_t idx_outer;
	// Array initialization.
	
	_nassert(((uint32_t)interfDetectArr % 8U) == 0);
	_nassert(((uint32_t)inp % 8U) == 0);
	for (idx_outer = 0; idx_outer < len; idx_outer += 1)
	{
		if (interfDetectArr[idx_outer])
		{
			inp[idx_outer].real = 0;
			inp[idx_outer].imag = 0;
		}
	}

	return 1;
}

/**
*  @b Description
*  @n
*    Approximation for the absolute value of a complex number.
*
*  @retval
*      Approximate absolute value of a complex number
*/	
int16_t jplApprox(cmplx16ImRe_t xl_inp)
{
	int16_t real = xl_inp.real;
	int16_t imag = xl_inp.imag;
	int16_t absValue;
	int16_t max, min;

	if (real < 0)
	{
		real = -real;
	}

	if (imag < 0)
	{
		imag = -imag;
	}

	max = imag;
	min = real;

	if (real > imag)
	{
		max = real;
		min = imag;
	}

	absValue = max + ((min * 3U) >> 3U);

	return absValue;
}
