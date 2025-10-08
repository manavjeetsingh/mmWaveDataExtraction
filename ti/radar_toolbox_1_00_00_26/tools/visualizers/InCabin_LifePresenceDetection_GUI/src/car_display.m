% Display cartoon images of occupants in various zone positions
function car_display(figureTab, tracker, numRows, xscale, seat_pos, frameIdx)

    global car_bg;
    global person;

    xstart = (1.0 - xscale) / 2;
    frameFlag = bitand(frameIdx, 3);

    ha = axes('parent', figureTab, 'units', 'normalized', 'position', [xstart 0 xscale 1]);


    if (0)
        % This creates the 'background' axes
        ha = axes('units','normalized', 'position',[0 0 1 1]);
        % Move the background axes to the bottom
        uistack(ha,'bottom');
        % Load in a background image and display it using the correct colors
        hi = imagesc(car_bg);

        %colormap gray
        % Turn the handlevisibility off so that we don't inadvertently plot into the axes again
        % Also, make the axes invisible
        set(ha,'handlevisibility','off', 'visible','off');
        hold on;
    end

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
               imshow(person, 'Border','tight');
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
               imshow(person, 'Border','tight');
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
               imshow(person, 'Border','tight');
            end
        end         
    else
        subplot('Position', seat_pos(1,:)); 
    end
hold off;
return
    