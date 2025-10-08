function  classStr = classification_logic_perSeat(Params, classStr, point3d, zoneMap, tracker)

if (isempty(point3d))
    return;
end

for idx = 1:Params.numZones   

    if (tracker(idx).state)
        classStr.frameCount(idx) = classStr.frameCount(idx) + 1;  
        classStr.xVol{idx} = [classStr.xVol{idx}, point3d(1,  1 ==  zoneMap(:, idx))];
        classStr.yVol{idx} = [classStr.yVol{idx}, point3d(2,  1 ==  zoneMap(:, idx))];
        classStr.zVol{idx} = [classStr.zVol{idx}, point3d(3,  1 ==  zoneMap(:, idx))];
        classStr.snrAcc(idx) = classStr.snrAcc(idx) + sum(point3d(4, 1 == zoneMap(:,idx)));

        if classStr.frameCount(idx) == Params.classParam.numFrameAvg 

            classStr.decision(idx) = 0;
            x_vol = classStr.xVol{idx};
            y_vol = classStr.yVol{idx};
            z_vol = classStr.zVol{idx};
            snr = classStr.snrAcc(idx) / Params.classParam.numFrameAvg; 
            %[void, volume] = convhull(double(classStr.xVol{idx}), double(classStr.yVol{idx}), double(classStr.zVol{idx}));
            volume = (std(x_vol))^2 + (std(y_vol))^2 + (std(z_vol))^2;
            display([idx, 10*log10(snr), volume]);
            if((10*log10(snr) < Params.classParam.thresh_snr(idx)) && (volume < Params.classParam.thresh_vol(idx)))
              classStr.decision(idx) = 1; %%1 means child and 2 means adult
            else
              classStr.decision(idx) = 2;
            end
            classStr = resetStat(classStr, idx);              
        end
    elseif (classStr.frameCount(idx) ~= 0)
         classStr = resetStat(classStr, idx);
         classStr.decision(idx) = 0;
    end           
end
end

function classStr = resetStat(classStr, idx)
    % classification structure reset
    classStr.frameCount(idx) = 0;
    classStr.xVol{idx} = [];
    classStr.yVol{idx} = [];
    classStr.zVol{idx} = [];        
    classStr.snrAcc(idx) = 0;
end