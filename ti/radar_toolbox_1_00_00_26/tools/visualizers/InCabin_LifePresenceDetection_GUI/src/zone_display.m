function zone_display(figureTab, zone, xLoc, yLoc, zLoc, row)

    AZ = -30; EL = 20; 
    clr = ['c', 'g', 'g', 'm', 'y','m','g','c'];
    
    %% front row
    %figure(figureID);
    axes('parent', figureTab);
    subplot(1, 1, 1);
    cla;
    view(AZ,EL);

    if (row == 1)
       % front driver and passenger side
       hold on;
       for zIdx = 1:2
            cZone = zone(zIdx);
            for cIdx = 1:cZone.numCuboids
                plotCuboids(cZone.cuboid(cIdx).x, cZone.cuboid(cIdx).y, cZone.cuboid(cIdx).z, clr(zIdx));
            end
       end

       axis([-1.5, 1.5, -0.2, 3.3, 0, 1.5])
       xlabel('x');
       ylabel('y');
       zlabel('z');
       scatter3(xLoc, yLoc, zLoc, 'filled','bo');
       title('First row seats')
       hold off;

    elseif (row == 2)
       % second row driver, middle and passenger side
       hold on;
       for zIdx = 3:5
           cZone = zone(zIdx);
           for cIdx = 1:cZone.numCuboids
               plotCuboids(cZone.cuboid(cIdx).x, cZone.cuboid(cIdx).y, cZone.cuboid(cIdx).z, clr(zIdx));
           end
       end

       axis([-1.5, 1.5, -0.2, 3.3, 0, 1.5])
       xlabel('x');
       ylabel('y');
       zlabel('z');
       scatter3(xLoc, yLoc, zLoc, 'filled','bo');
       title('Second row seats')
       hold off;
    elseif (row == 3)
       hold on;
       for zIdx = 6:8
           cZone = zone(zIdx);
           for cIdx = 1:cZone.numCuboids
               plotCuboids(cZone.cuboid(cIdx).x, cZone.cuboid(cIdx).y, cZone.cuboid(cIdx).z, clr(zIdx));
           end
       end

       axis([-1.5, 1.5, -0.2, 3.3, 0, 1.5])
       xlabel('x');
       ylabel('y');
       zlabel('z');
       scatter3(xLoc, yLoc, zLoc, 'filled','bo');
       title('Second row seats')
       hold off;        
    end
end


function plotCuboids(xin, yin, zin, color)

    a = -pi : pi/2 : pi;                      % Define Corners
    ph = pi/4;

    t1 = (xin(1) + xin(2))/2;  %min and max X
    t2 = (xin(2) - xin(1))/2;
    x = t1 + t2*[cos(a+ph); cos(a+ph)]/cos(ph);

    t1 = (yin(1) + yin(2))/2;  %min and max Y
    t2 = (yin(2) - yin(1))/2;
    y = t1 + t2*[sin(a+ph); sin(a+ph)]/sin(ph);

    t1 = (zin(1) + zin(2))/2;  %min and max Z
    t2 = (zin(2) - zin(1))/2;
    z = t1 + t2*[-ones(size(a)); ones(size(a))];

    surf(x, y, z, 'FaceColor',color);
%    hold on;
end