function [occCount, tracker] = occupancyDetection(numZones, tracker, point3D, zoneMap, smParams)

    occCount = 0;

    for idx = 1:numZones
        % Find the number of detected points in this zone
        tracker(idx).numPoints = sum(zoneMap(:,idx));

        % Calculate the average SNR for detected points in this zone
        if (tracker(idx).numPoints > 0)
            tracker(idx).avgSnr = mean(point3D(4, 1 == zoneMap(:,idx)));
        else
            tracker(idx).avgSnr = 0.0;
        end
    end
      
    % Check for overload conditions (large movements in the row)
    for idx = 1:numZones
        if (tracker(idx).avgSnr >= smParams{1}.overloadThreshold)

            % If an overload occurs, freeze all zones as well.
            for idx2 = 1: numZones
                tracker(idx2).freeze = 2;   % freeze for 2 frames
            end            
        elseif (tracker(idx).freeze > 0)
            tracker(idx).freeze = tracker(idx).freeze - 1;
        end
    end

    % Update the occupancy state of each zone
    for idx = 1:numZones  
        zoneType = tracker(idx).zoneType + 1; 
        tracker = updateTrackerStateMachine(idx, tracker, smParams{zoneType});
        
        if (tracker(idx).state == 1)
            occCount = occCount + 1;
        end
    end
end


function tracker = updateTrackerStateMachine(id, tracker, smParams)
    trackerCur = tracker(id);
    neighbors =  tracker(id).neighbors; 
    if (trackerCur.freeze == 0) 
        maxAvgSnr = smParams.avgSnrForEnterThreshold2; 
        for i = 1:length(neighbors)
            tt = neighbors(i);
            avgSnrNeighbor = tracker(tt).avgSnr; 
            maxAvgSnr = max(maxAvgSnr, avgSnrNeighbor);
        end

        switch trackerCur.state
            case 0 % NOT_OCCUPIED
                if (((trackerCur.numPoints > smParams.numPointForEnterThreshold1) && (trackerCur.avgSnr > smParams.avgSnrForEnterThreshold1)) || ...
                    ((trackerCur.numPoints > smParams.numPointForEnterThreshold2) && (trackerCur.avgSnr > maxAvgSnr)))
                    trackerCur.numEntryCount = trackerCur.numEntryCount + 1; 
                else
                    trackerCur.numEntryCount = 0;  %%reset the counter
                end

                if(trackerCur.numEntryCount >= smParams.numEntryThreshold) 
                    trackerCur.state = 1;
                    trackerCur.detect2freeCount = 0;
                end                    
    
            case 1 % OCCUPIED
                if ((trackerCur.numPoints > smParams.numPointForStayThreshold) && (trackerCur.avgSnr > smParams.avgSnrForStayThreshold))
                    % still detected
                    trackerCur.detect2freeCount = 0;
                elseif (trackerCur.numPoints < smParams.numPointToForget)
                    % Miss
                    if (trackerCur.detect2freeCount > smParams.forgetThreshold)
                        trackerCur.state = 0;
                    else
                        trackerCur.detect2freeCount = trackerCur.detect2freeCount + 1;
                    end
                else
                    trackerCur.detect2freeCount = trackerCur.detect2freeCount - 1;
                end                                
        end
    end
    tracker(id) = trackerCur;
end



