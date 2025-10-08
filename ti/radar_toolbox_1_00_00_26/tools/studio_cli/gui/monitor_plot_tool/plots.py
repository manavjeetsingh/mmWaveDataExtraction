import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import json
import os

SG_WINDOW_SIZE=11
SG_POLY_ORDER=1

gridspec_kw = {
                    'left' : 0.10,
                    'bottom' : 0.13,
                    'right' : 0.95,
                    'top' : 0.88,
                    'wspace' : 0.45,
                    'hspace' : 0.55,
                    }

gridspec_kw_2 = {
                    'top':0.898,
                    'bottom':0.168,
                    'left':0.203,
                    'right':0.918,
                    'hspace':0.2,
                    'wspace':0.597
                }

def savgol_filter_wrapper(data, window_size, poly_size):
    if len(data) < window_size:
        return data
    
    if len(data) <poly_size:
        return data
        
    return savgol_filter(data, window_size, poly_size)

def rlMonTempReportDataPlot(path): 
 
    TEMP_RX0 = []
    TEMP_RX1 = [] 
    TEMP_RX2 = []
    TEMP_RX3 = []
    TEMP_TX0 = []
    TEMP_TX1 = []
    TEMP_TX2 = []
    TEMP_PM =  []
    TEMP_DIG1 = []
    TEMP_DIG2 = []
    
    
    with open(os.path.join(path, 'rlMonTempReportData.json'), 'r') as f:
        data = json.loads(f.read())
    for report in data['data']:
        tempValues = report['tempValues']
        
        TEMP_RX0.append(tempValues[0])
        TEMP_RX1.append(tempValues[1]) 
        TEMP_RX2.append(tempValues[2])
        TEMP_RX3.append(tempValues[3])
        TEMP_TX0.append(tempValues[4])
        TEMP_TX1.append(tempValues[5])
        TEMP_TX2.append(tempValues[6])
        TEMP_PM.append(tempValues[7])
        TEMP_DIG1.append(tempValues[8])
        TEMP_DIG2.append(tempValues[9])
    

    fig, axs = plt.subplots(2, 5, gridspec_kw=gridspec_kw, num='Radar Temperature Monitor', figsize=(13, 7))
    
    for i in range (0, 9):
        ax = axs[i//5][i%5]
        ax.clear()
        ax.set_xlabel('Time (FTTI units)')
        ax.set_ylabel('Temp (C)')

    
    ax = axs[0][0]
    ax.set_title('TEMP_RX0')
    ax.plot(savgol_filter_wrapper(TEMP_RX0, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][1]
    ax.set_title('TEMP_RX1')
    ax.plot(savgol_filter_wrapper(TEMP_RX1, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][2]
    ax.set_title('TEMP_RX2')
    ax.plot(savgol_filter_wrapper(TEMP_RX2, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][3]
    ax.set_title('TEMP_RX3')
    ax.plot(savgol_filter_wrapper(TEMP_RX3, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][4]
    ax.set_title('TEMP_TX0')
    ax.plot(savgol_filter_wrapper(TEMP_TX0, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][0]
    ax.set_title('TEMP_TX1')
    ax.plot(savgol_filter_wrapper(TEMP_TX1, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][1]
    ax.set_title('TEMP_TX2')
    ax.plot(savgol_filter_wrapper(TEMP_TX2, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][2]
    ax.set_title('TEMP_DIG1')
    ax.plot(savgol_filter_wrapper(TEMP_DIG1, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][3]
    ax.set_title('TEMP_DIG2')
    ax.plot(savgol_filter_wrapper(TEMP_DIG2, SG_WINDOW_SIZE, SG_POLY_ORDER))

    plt.draw()
    plt.pause(0.001)
    
def rlMonTxPowRepDataPlot(path):
   
    TX_POWER_VALUE_RF1_TX0 = []
    TX_POWER_VALUE_RF2_TX0 = []
    TX_POWER_VALUE_RF3_TX0 = []
    TX_POWER_VALUE_RF1_TX1 = []
    TX_POWER_VALUE_RF2_TX1 = []
    TX_POWER_VALUE_RF3_TX1 = []
    TX_POWER_VALUE_RF1_TX2 = []
    TX_POWER_VALUE_RF2_TX2 = []
    TX_POWER_VALUE_RF3_TX2 = []
    
    with open(os.path.join(path, 'rlMonTxPowRepData0.json'), 'r') as f:
        data = json.loads(f.read())
    for report in data['data']:
        txPowVal = report['txPowVal']
        TX_POWER_VALUE_RF1_TX0.append(txPowVal[0]/10)
        TX_POWER_VALUE_RF2_TX0.append(txPowVal[1]/10) 
        TX_POWER_VALUE_RF3_TX0.append(txPowVal[2]/10)

    with open(os.path.join(path, 'rlMonTxPowRepData1.json'), 'r') as f:
        data = json.loads(f.read())
    for report in data['data']:
        txPowVal = report['txPowVal']
        TX_POWER_VALUE_RF1_TX1.append(txPowVal[0]/10)
        TX_POWER_VALUE_RF2_TX1.append(txPowVal[1]/10) 
        TX_POWER_VALUE_RF3_TX1.append(txPowVal[2]/10)
    
    try:
        with open(os.path.join(path, 'rlMonTxPowRepData2.json'), 'r') as f:
            data = json.loads(f.read())
        for report in data['data']:
            txPowVal = report['txPowVal']
            TX_POWER_VALUE_RF1_TX2.append(txPowVal[0]/10)
            TX_POWER_VALUE_RF2_TX2.append(txPowVal[1]/10) 
            TX_POWER_VALUE_RF3_TX2.append(txPowVal[2]/10)
    except:
        pass
    
    
    fig, axs = plt.subplots(3, 3, gridspec_kw=gridspec_kw, num='Radar Tx Power Monitor', figsize=(10, 8))
    
    for i in range (0, 9):
        ax = axs[i//3][i%3]
        ax.clear()
        ax.set_xlabel('Time (FTTI units)')
        ax.set_ylabel('Power (dBm)')
        ax.set_ylim([9,15])

    
    ax = axs[0][0]
    ax.set_title('TX_POWER_VALUE_RF1_TX0')
    ax.plot(savgol_filter_wrapper(TX_POWER_VALUE_RF1_TX0, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][1]
    ax.set_title('TX_POWER_VALUE_RF2_TX0')
    ax.plot(savgol_filter_wrapper(TX_POWER_VALUE_RF2_TX0, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][2]
    ax.set_title('TX_POWER_VALUE_RF3_TX0')
    ax.plot(savgol_filter_wrapper(TX_POWER_VALUE_RF3_TX0, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][0]
    ax.set_title('TX_POWER_VALUE_RF1_TX1')
    ax.plot(savgol_filter_wrapper(TX_POWER_VALUE_RF1_TX1, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][1]
    ax.set_title('TX_POWER_VALUE_RF2_TX1')
    ax.plot(savgol_filter_wrapper(TX_POWER_VALUE_RF2_TX1, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][2]
    ax.set_title('TX_POWER_VALUE_RF3_TX1')
    ax.plot(savgol_filter_wrapper(TX_POWER_VALUE_RF3_TX1, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[2][0]
    ax.set_title('TX_POWER_VALUE_RF1_TX2')
    ax.plot(savgol_filter_wrapper(TX_POWER_VALUE_RF1_TX2, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[2][1]
    ax.set_title('TX_POWER_VALUE_RF2_TX2')
    ax.plot(savgol_filter_wrapper(TX_POWER_VALUE_RF2_TX2, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[2][2]
    ax.set_title('TX_POWER_VALUE_RF3_TX2')
    ax.plot(savgol_filter_wrapper(TX_POWER_VALUE_RF3_TX2, SG_WINDOW_SIZE, SG_POLY_ORDER))
    


    plt.draw()
    plt.pause(0.001)
    
def rlMonPllConVoltRepDataPlot(path):
    APLL_VCTRL = []
    SYNTH_VCO1_VCTRL_MAX_FREQ = [] 
    SYNTH_VCO1_VCTRL_MIN_FREQ = []
    SYNTH_VCO1_SLOPE  = []
    SYNTH_VCO2_VCTRL_MAX_FREQ = []
    SYNTH_VCO2_VCTRL_MIN_FREQ = []
    SYNTH_VCO2_SLOPE  = []
    
    
    with open(os.path.join(path, 'rlMonPllConVoltRepData.json'), 'r') as f:
        data = json.loads(f.read())
    for report in data['data']:
        pllContVoltVal = report['pllContVoltVal']
        
        APLL_VCTRL.append(pllContVoltVal[0])
        SYNTH_VCO1_VCTRL_MAX_FREQ.append(pllContVoltVal[1])
        SYNTH_VCO1_VCTRL_MIN_FREQ.append(pllContVoltVal[2])
        SYNTH_VCO1_SLOPE.append(pllContVoltVal[3])
        SYNTH_VCO2_VCTRL_MAX_FREQ.append(pllContVoltVal[4])
        SYNTH_VCO2_VCTRL_MIN_FREQ.append(pllContVoltVal[5])
        SYNTH_VCO2_SLOPE.append(pllContVoltVal[6])
    

    fig, axs = plt.subplots(2, 4, gridspec_kw=gridspec_kw, num='Radar PLL Control Voltage Monitor', figsize=(14, 8))
    
    for i in range (0, 7):
        ax = axs[i//4][i%4]
        ax.clear()
        ax.set_xlabel('Time (FTTI units)')

    
    ax = axs[0][0]
    ax.set_title('APLL_VCTRL')
    ax.set_ylabel('mV')
    ax.plot(savgol_filter_wrapper(APLL_VCTRL, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][1]
    ax.set_title('SYNTH_VCO1_VCTRL_MAX_FREQ')
    ax.set_ylabel('mV')
    ax.plot(savgol_filter_wrapper(SYNTH_VCO1_VCTRL_MAX_FREQ, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][2]
    ax.set_title('SYNTH_VCO1_VCTRL_MIN_FREQ')
    ax.set_ylabel('mV')
    ax.plot(savgol_filter_wrapper(SYNTH_VCO1_VCTRL_MIN_FREQ, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][3]
    ax.set_title('SYNTH_VCO1_SLOPE')
    ax.set_ylabel('MHz/V')
    ax.plot(savgol_filter_wrapper(SYNTH_VCO1_SLOPE, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][0]
    ax.set_title('SYNTH_VCO2_VCTRL_MAX_FREQ')
    ax.set_ylabel('mV')
    ax.plot(savgol_filter_wrapper(SYNTH_VCO2_VCTRL_MAX_FREQ, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][1]
    ax.set_title('SYNTH_VCO2_VCTRL_MIN_FREQ')
    ax.set_ylabel('mV')
    ax.plot(savgol_filter_wrapper(SYNTH_VCO2_VCTRL_MIN_FREQ, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][2]
    ax.set_title('SYNTH_VCO2_SLOPE')
    ax.set_ylabel('MHz/V')
    ax.plot(savgol_filter_wrapper(SYNTH_VCO2_SLOPE, SG_WINDOW_SIZE, SG_POLY_ORDER))

    plt.draw()
    plt.pause(0.001)
    
def rlMonTxBallBreakRepDataPlot(path):
    TX_REFL_COEFF_VALUE_TX0 = []
    TX_REFL_COEFF_VALUE_TX1 = []
    TX_REFL_COEFF_VALUE_TX2 = []
    
    with open(os.path.join(path, 'rlMonTxBallBreakRepData0.json'), 'r') as f:
        data = json.loads(f.read())
    for report in data['data']:
        txReflCoefVal = report['txReflCoefVal']
        TX_REFL_COEFF_VALUE_TX0.append(txReflCoefVal/10)

    with open(os.path.join(path, 'rlMonTxBallBreakRepData1.json'), 'r') as f:
        data = json.loads(f.read())
    for report in data['data']:
        txReflCoefVal = report['txReflCoefVal']
        TX_REFL_COEFF_VALUE_TX1.append(txReflCoefVal/10)
    
    try:
        with open(os.path.join(path, 'rlMonTxBallBreakRepData2.json'), 'r') as f:
            data = json.loads(f.read())
        for report in data['data']:
            txReflCoefVal = report['txReflCoefVal']
            TX_REFL_COEFF_VALUE_TX2.append(txReflCoefVal/10)
    except:
        pass
    
    
    fig, axs = plt.subplots(1, 3, gridspec_kw=gridspec_kw, num='Radar Tx Ballbreak Monitor', figsize=(8.5, 3.5))
    
    for i in range (0, 3):
        ax = axs[i]
        ax.clear()
        ax.set_xlabel('Time (FTTI units)')
        ax.set_ylabel('Power (dB)')

    
    ax = axs[0]
    ax.set_title('TX_REFL_COEFF_VALUE_TX0')
    ax.plot(savgol_filter_wrapper(TX_REFL_COEFF_VALUE_TX0, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1]
    ax.set_title('TX_REFL_COEFF_VALUE_TX1')
    ax.plot(savgol_filter_wrapper(TX_REFL_COEFF_VALUE_TX1, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[2]
    ax.set_title('TX_REFL_COEFF_VALUE_TX2')
    ax.plot(savgol_filter_wrapper(TX_REFL_COEFF_VALUE_TX2, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    
    


    plt.draw()
    plt.pause(0.001)
    
def rlMonGpadcIntAnaSigRepDataPlot(path):
    GPADC_REF1_VALUE = []
    GPADC_REF2_VALUE = []
    
    with open(os.path.join(path, 'rlMonGpadcIntAnaSigRepData.json'), 'r') as f:
        data = json.loads(f.read())
    for report in data['data']:
        gpadcRef1Val = report['gpadcRef1Val']
        gpadcRef2Val = report['gpadcRef2Val']
        GPADC_REF1_VALUE.append(gpadcRef1Val*1.8/1000)
        GPADC_REF2_VALUE.append(gpadcRef2Val*1.8/1000)
        
    
    
    fig, axs = plt.subplots(1, 2, gridspec_kw=gridspec_kw_2, num='GPADC Monitor', figsize=(8, 3.5))
    
    for i in range (0, 2):
        ax = axs[i]
        ax.clear()
        ax.set_xlabel('Time (FTTI units)')
        ax.set_ylabel('Volts')

    
    ax = axs[0]
    ax.set_title('GPADC_REF1_VALUE')
    ax.plot(savgol_filter_wrapper(GPADC_REF1_VALUE, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1]
    ax.set_title('GPADC_REF2_VALUE')
    ax.plot(savgol_filter_wrapper(GPADC_REF2_VALUE, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    plt.draw()
    plt.pause(0.001)
    
def rlMonRxIfStageRepDataPlot(path):
    HPF_CUTOFF_FREQ_ERROR_VALUE = {'I':[[], [], [], []], 'Q':[[], [], [], []]}
    LPF_CUTOFF_FREQ_ERROR_VALUE = {'I':[[], [], [], []], 'Q':[[], [], [], []]}
    RX_IFA_GAIN_ERROR_VALUE = {'I':[[], [], [], []], 'Q':[[], [], [], []]}

    
    with open(os.path.join(path, 'rlMonRxIfStageRepData.json'), 'r') as f:
        data = json.loads(f.read())
    for report in data['data']:
        hpfCutOffFreqEr = report['hpfCutOffFreqEr']
        HPF_CUTOFF_FREQ_ERROR_VALUE['I'][0].append(hpfCutOffFreqEr[0])
        HPF_CUTOFF_FREQ_ERROR_VALUE['I'][1].append(hpfCutOffFreqEr[1])
        HPF_CUTOFF_FREQ_ERROR_VALUE['I'][2].append(hpfCutOffFreqEr[2])
        HPF_CUTOFF_FREQ_ERROR_VALUE['I'][3].append(hpfCutOffFreqEr[3])
        HPF_CUTOFF_FREQ_ERROR_VALUE['Q'][0].append(hpfCutOffFreqEr[4])
        HPF_CUTOFF_FREQ_ERROR_VALUE['Q'][1].append(hpfCutOffFreqEr[5])
        HPF_CUTOFF_FREQ_ERROR_VALUE['Q'][2].append(hpfCutOffFreqEr[6])
        HPF_CUTOFF_FREQ_ERROR_VALUE['Q'][3].append(hpfCutOffFreqEr[7])
        
        lpfCutOffFreqEr = report['lpfCutOffFreqEr']
        LPF_CUTOFF_FREQ_ERROR_VALUE['I'][0].append(lpfCutOffFreqEr[0])
        LPF_CUTOFF_FREQ_ERROR_VALUE['I'][1].append(lpfCutOffFreqEr[1])
        LPF_CUTOFF_FREQ_ERROR_VALUE['I'][2].append(lpfCutOffFreqEr[2])
        LPF_CUTOFF_FREQ_ERROR_VALUE['I'][3].append(lpfCutOffFreqEr[3])
        LPF_CUTOFF_FREQ_ERROR_VALUE['Q'][0].append(lpfCutOffFreqEr[4])
        LPF_CUTOFF_FREQ_ERROR_VALUE['Q'][1].append(lpfCutOffFreqEr[5])
        LPF_CUTOFF_FREQ_ERROR_VALUE['Q'][2].append(lpfCutOffFreqEr[6])
        LPF_CUTOFF_FREQ_ERROR_VALUE['Q'][3].append(lpfCutOffFreqEr[7])
        
        rxIfaGainErVal = report['rxIfaGainErVal']
        RX_IFA_GAIN_ERROR_VALUE['I'][0].append(rxIfaGainErVal[0]/10)
        RX_IFA_GAIN_ERROR_VALUE['I'][1].append(rxIfaGainErVal[1]/10)
        RX_IFA_GAIN_ERROR_VALUE['I'][2].append(rxIfaGainErVal[2]/10)
        RX_IFA_GAIN_ERROR_VALUE['I'][3].append(rxIfaGainErVal[3]/10)
        RX_IFA_GAIN_ERROR_VALUE['Q'][0].append(rxIfaGainErVal[4]/10)
        RX_IFA_GAIN_ERROR_VALUE['Q'][1].append(rxIfaGainErVal[5]/10)
        RX_IFA_GAIN_ERROR_VALUE['Q'][2].append(rxIfaGainErVal[6]/10)
        RX_IFA_GAIN_ERROR_VALUE['Q'][3].append(rxIfaGainErVal[7]/10)
    
    
    fig, axs = plt.subplots(3, 4, gridspec_kw=gridspec_kw, num='IFStage Monitor', figsize=(16, 10))
    
    for i in range (0, 8):
        ax = axs[i//4][i%4]
        ax.clear()
        ax.set_xlabel('Time (FTTI units)')
        ax.set_ylabel('Error (%)')
        
    for i in range (8, 12):
        ax = axs[i//4][i%4]
        ax.clear()
        ax.set_xlabel('Time (FTTI units)')
        ax.set_ylabel('Error (dB)')

    
    ax = axs[0][0]
    ax.set_title('HPF_CUTOFF_FREQ_ERROR_VALUE RX0 I')
    ax.plot(savgol_filter_wrapper(HPF_CUTOFF_FREQ_ERROR_VALUE['I'][0], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[0][1]
    ax.set_title('HPF_CUTOFF_FREQ_ERROR_VALUE RX1 I')
    ax.plot(savgol_filter_wrapper(HPF_CUTOFF_FREQ_ERROR_VALUE['I'][1], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[0][2]
    ax.set_title('HPF_CUTOFF_FREQ_ERROR_VALUE RX2 I')
    ax.plot(savgol_filter_wrapper(HPF_CUTOFF_FREQ_ERROR_VALUE['I'][2], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[0][3]
    ax.set_title('HPF_CUTOFF_FREQ_ERROR_VALUE RX3 I')
    ax.plot(savgol_filter_wrapper(HPF_CUTOFF_FREQ_ERROR_VALUE['I'][3], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[0][0]
    ax.set_title('HPF_CUTOFF_FREQ_ERROR_VALUE RX0')
    ax.plot(savgol_filter_wrapper(HPF_CUTOFF_FREQ_ERROR_VALUE['Q'][0], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    ax = axs[0][1]
    ax.set_title('HPF_CUTOFF_FREQ_ERROR_VALUE RX1')
    ax.plot(savgol_filter_wrapper(HPF_CUTOFF_FREQ_ERROR_VALUE['Q'][1], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    ax = axs[0][2]
    ax.set_title('HPF_CUTOFF_FREQ_ERROR_VALUE RX2')
    ax.plot(savgol_filter_wrapper(HPF_CUTOFF_FREQ_ERROR_VALUE['Q'][2], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    ax = axs[0][3]
    ax.set_title('HPF_CUTOFF_FREQ_ERROR_VALUE RX3')
    ax.plot(savgol_filter_wrapper(HPF_CUTOFF_FREQ_ERROR_VALUE['Q'][3], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    
    
    
    ax = axs[1][0]
    ax.set_title('LPF_CUTOFF_FREQ_ERROR_VALUE RX0 I')
    ax.plot(savgol_filter_wrapper(LPF_CUTOFF_FREQ_ERROR_VALUE['I'][0], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[1][1]
    ax.set_title('LPF_CUTOFF_FREQ_ERROR_VALUE RX1 I')
    ax.plot(savgol_filter_wrapper(LPF_CUTOFF_FREQ_ERROR_VALUE['I'][1], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[1][2]
    ax.set_title('LPF_CUTOFF_FREQ_ERROR_VALUE RX2 I')
    ax.plot(savgol_filter_wrapper(LPF_CUTOFF_FREQ_ERROR_VALUE['I'][2], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[1][3]
    ax.set_title('LPF_CUTOFF_FREQ_ERROR_VALUE RX3 I')
    ax.plot(savgol_filter_wrapper(LPF_CUTOFF_FREQ_ERROR_VALUE['I'][3], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[1][0]
    ax.set_title('LPF_CUTOFF_FREQ_ERROR_VALUE RX0')
    ax.plot(savgol_filter_wrapper(LPF_CUTOFF_FREQ_ERROR_VALUE['Q'][0], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    ax = axs[1][1]
    ax.set_title('LPF_CUTOFF_FREQ_ERROR_VALUE RX1')
    ax.plot(savgol_filter_wrapper(LPF_CUTOFF_FREQ_ERROR_VALUE['Q'][1], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    ax = axs[1][2]
    ax.set_title('LPF_CUTOFF_FREQ_ERROR_VALUE RX2')
    ax.plot(savgol_filter_wrapper(LPF_CUTOFF_FREQ_ERROR_VALUE['Q'][2], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    ax = axs[1][3]
    ax.set_title('LPF_CUTOFF_FREQ_ERROR_VALUE RX3')
    ax.plot(savgol_filter_wrapper(LPF_CUTOFF_FREQ_ERROR_VALUE['Q'][3], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    
    
    
    ax = axs[2][0]
    ax.set_title('RX_IFA_GAIN_ERROR_VALUE RX0 I')
    ax.plot(savgol_filter_wrapper(RX_IFA_GAIN_ERROR_VALUE['I'][0], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[2][1]
    ax.set_title('RX_IFA_GAIN_ERROR_VALUE RX1 I')
    ax.plot(savgol_filter_wrapper(RX_IFA_GAIN_ERROR_VALUE['I'][1], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[2][2]
    ax.set_title('RX_IFA_GAIN_ERROR_VALUE RX2 I')
    ax.plot(savgol_filter_wrapper(RX_IFA_GAIN_ERROR_VALUE['I'][2], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[2][3]
    ax.set_title('RX_IFA_GAIN_ERROR_VALUE RX3 I')
    ax.plot(savgol_filter_wrapper(RX_IFA_GAIN_ERROR_VALUE['I'][3], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='--', label='I')
    
    ax = axs[2][0]
    ax.set_title('RX_IFA_GAIN_ERROR_VALUE RX0')
    ax.plot(savgol_filter_wrapper(RX_IFA_GAIN_ERROR_VALUE['Q'][0], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    ax = axs[2][1]
    ax.set_title('RX_IFA_GAIN_ERROR_VALUE RX1')
    ax.plot(savgol_filter_wrapper(RX_IFA_GAIN_ERROR_VALUE['Q'][1], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    ax = axs[2][2]
    ax.set_title('RX_IFA_GAIN_ERROR_VALUE RX2')
    ax.plot(savgol_filter_wrapper(RX_IFA_GAIN_ERROR_VALUE['Q'][2], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    ax = axs[2][3]
    ax.set_title('RX_IFA_GAIN_ERROR_VALUE RX3')
    ax.plot(savgol_filter_wrapper(RX_IFA_GAIN_ERROR_VALUE['Q'][3], SG_WINDOW_SIZE, SG_POLY_ORDER), linestyle='-', alpha=0.7, label='Q')
    
    for i in range (0, 12):
        ax = axs[i//4][i%4]
        ax.legend()
    
    
    plt.draw()
    plt.pause(0.001)

def rlMonSynthFreqRepDataPlot(path):
    MAX_FREQUENCY_ERROR_VALUE = []
    FREQUENCY_FAILURE_COUNT = []
    
    with open(os.path.join(path, 'rlMonSynthFreqRepData.json'), 'r') as f:
        data = json.loads(f.read())
    for report in data['data']:
        maxFreqErVal = report['maxFreqErVal']
        freqFailCnt = report['freqFailCnt']
        MAX_FREQUENCY_ERROR_VALUE.append(maxFreqErVal/1000)
        FREQUENCY_FAILURE_COUNT.append(freqFailCnt/1000)
        
    
    
    fig, axs = plt.subplots(1, 2, gridspec_kw=gridspec_kw_2, num='Synth Freq Monitor', figsize=(8, 3.5))
    
    ax = axs[0]
    ax.clear()
    ax.set_xlabel('Time (FTTI units)')
    ax.set_ylabel('mHz')
    
    ax = axs[1]
    ax.clear()
    ax.set_xlabel('Time (FTTI units)')
    ax.set_ylabel('Count')

    
    ax = axs[0]
    ax.set_title('MAX_FREQUENCY_ERROR_VALUE')
    ax.plot(MAX_FREQUENCY_ERROR_VALUE)
    
    ax = axs[1]
    ax.set_title('FREQUENCY_FAILURE_COUNT')
    ax.plot(FREQUENCY_FAILURE_COUNT)
    
    plt.draw()
    plt.pause(0.001)
     
def rlMonDccClkFreqRepDataPlot(path):
    BSS_600M  = []
    BSS_200M = [] 
    BSS_100M = []
    GPADC_10M  = []
    RCOSC_10M = []
    RAMPGEN_100M = []
    
    
    with open(os.path.join(path, 'rlMonDccClkFreqRepData.json'), 'r') as f:
        data = json.loads(f.read())
    for report in data['data']:
        freqMeasVal = report['freqMeasVal']
        
        BSS_600M.append(freqMeasVal[0]*0.1)
        BSS_200M.append(freqMeasVal[1]*0.1)
        BSS_100M.append(freqMeasVal[2]*0.1)
        GPADC_10M.append(freqMeasVal[3]*0.1)
        RCOSC_10M.append(freqMeasVal[4]*0.1)
        RAMPGEN_100M.append(freqMeasVal[5]*0.1)
    

    fig, axs = plt.subplots(2, 3, gridspec_kw=gridspec_kw, num='Radar DCC Monitor', figsize=(12, 8))
    
    for i in range (0, 6):
        ax = axs[i//3][i%3]
        ax.clear()
        ax.set_xlabel('Time (FTTI units)')
        ax.set_ylabel('Freq (mHz)')

    
    ax = axs[0][0]
    ax.set_title('BSS_600M')
    ax.plot(savgol_filter_wrapper(BSS_600M, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][1]
    ax.set_title('BSS_200M')
    ax.plot(savgol_filter_wrapper(BSS_200M, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[0][2]
    ax.set_title('BSS_100M')
    ax.plot(savgol_filter_wrapper(BSS_100M, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][0]
    ax.set_title('GPADC_10M')
    ax.plot(savgol_filter_wrapper(GPADC_10M, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][1]
    ax.set_title('RCOSC_10M')
    ax.plot(savgol_filter_wrapper(RCOSC_10M, SG_WINDOW_SIZE, SG_POLY_ORDER))
    
    ax = axs[1][2]
    ax.set_title('RAMPGEN_100M')
    ax.plot(savgol_filter_wrapper(RAMPGEN_100M, SG_WINDOW_SIZE, SG_POLY_ORDER))

    plt.draw()
    plt.pause(0.001)