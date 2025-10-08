% Display cartoon images of occupants in various zone positions
function car_display(figureTab, tracker, classStr, numRows, xscale, seat_pos, frameIdx)

    global car_bg;
    global person;
    global child;
    global adult;

    xstart = (1.0 - xscale) / 2;

    ha = axes('parent', figureTab, 'units', 'normalized', 'position', [xstart 0 xscale 1]);

    frameFlag = bitand(frameIdx, 3);

    hold on;

    if (frameFlag == 1)
        %Front row
        for i = 1:2        
            subplot('Position', seat_pos(i,:));
            cla(subplot('Position', seat_pos(i,:)));
            set(gca,'visible', 'off', 'YTick',[],'XTick',[]);
            set(gca,'Color','none');

            xlabel('');
            ylabel('');

            if (tracker(i).state)
                switch classStr.decision(i)
                    case 0
                        imshow(person, 'Border','tight');
                    case 1
                        imshow(child, 'Border','tight');
                    case 2
                        imshow(adult, 'Border','tight');
                end 
            end
        end 

    elseif ((frameFlag == 2) && (numRows > 1))
        % second row if exist
        for i = 3:5        
            subplot('Position', seat_pos(i,:));
            cla(subplot('Position', seat_pos(i,:)));
            set(gca,'visible', 'off', 'YTick',[],'XTick',[]);
            set(gca,'Color','none');

            xlabel('');
            ylabel('');

            if (tracker(i).state)
                switch classStr.decision(i)
                    case 0
                        imshow(person, 'Border','tight');
                    case 1
                        imshow(child, 'Border','tight');
                    case 2
                        imshow(adult, 'Border','tight');
                end 
            end
        end 

    elseif ((frameFlag == 3) && (numRows > 2))
        % third row if exist
        for i = 6:8        
            subplot('Position', seat_pos(i,:));
            cla(subplot('Position', seat_pos(i,:)));
            set(gca,'visible', 'off', 'YTick',[],'XTick',[]);
            set(gca,'Color','none');

            xlabel('');
            ylabel('');

            if (tracker(i).state)  
                switch classStr.decision(i)
                    case 0
                        imshow(person, 'Border','tight');
                    case 1
                        imshow(child, 'Border','tight');
                    case 2
                        imshow(adult, 'Border','tight');
                end 
            end
        end         
    else
        subplot('Position', seat_pos(1,:)); 
    end
hold off;
return
    