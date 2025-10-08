function incabin_detectionDisplay(figureTab, xLoc, yLoc, zLoc, tracker, zone, intBound)
%color = 'g';
%color2 = 'c';
global totNumRows; 

%% front row
axes('parent', figureTab); 
subplot(1, 1, 1);
hold off;
cla;
scatter(xLoc, yLoc, 'm*','MarkerEdgeColor', 'b');
hold on;

grid on;
xlabel('x (m)');
ylabel('y (m)');

%Draw the interior boundary marked by the configuration
xlen = intBound.maxX - intBound.minX;
ylen = intBound.maxY - intBound.minY;
pos=[intBound.minX  intBound.minY  xlen  ylen];
R = rectangle('Position',pos,'Curvature',[0.1 0.1], 'EdgeColor', [1 0 1]);

if totNumRows < 3
    seatStr = {'Row 1 Driver','Row 1 Passenger','Row 2 Driver','Row 2 Middle','Row 2 Passngr','','','',''};
else
    seatStr = {'Row 1 Driver','Row 1 Passenger','Row 2 Driver','Row 2 Middle','Row 2 Passngr','Row 3 Driver','Row 3 Middle','Row 3 Passngr','','','',''};
end
for idx = 1:length(tracker)
    if (tracker(idx).state)
        pos=[zone(idx).x_start zone(idx).y_start zone(idx).x_len zone(idx).y_len];
        if (tracker(idx).freeze)
            R = rectangle('Position',pos,'Curvature',[0.1 0.1],'FaceColor', [.75 0  0 .3], 'EdgeColor', [1 0 0]);
        else
            R = rectangle('Position',pos,'Curvature',[0.1 0.1],'FaceColor', [.5 .5 .5 .3], 'EdgeColor', [0 1 0]);
        end
        text('Position',[zone(idx).x_start+zone(idx).x_len zone(idx).y_start + zone(idx).y_len/2],'string',seatStr{idx})
    end
end

axis([intBound.minX-0.5, intBound.maxX + 0.5, intBound.minY - 0.5, intBound.maxY + 0.5])
set(gca,'xdir','reverse','ydir','reverse')
hold off; 
