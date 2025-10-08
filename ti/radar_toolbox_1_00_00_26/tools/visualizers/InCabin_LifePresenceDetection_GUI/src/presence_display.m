% Display cartoon images of occupants in various zone positions
function presence_display(figureTab, tracker)

    global presence_icon;

    presence_pos = [0.33    0.30    0.3500    0.350];
    
    ha = axes('parent', figureTab, 'units', 'normalized', 'position', [0 0 1 1]);
    subplot('Position', presence_pos);
    cla(subplot('Position', presence_pos));
    set(gca,'visible', 'off', 'YTick',[],'XTick',[]);
    set(gca,'Color','none');
    xlabel('');
    ylabel('');

    hold on;
    %OccupiedFlag = 0; 
    for i = 1:length(tracker)
        if (tracker(i).state == 1)
            %OccupiedFlag = 1;
            imshow(presence_icon, 'Border','tight');
        end
    end    
    hold off;

return
    