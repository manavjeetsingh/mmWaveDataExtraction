 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 %
 %      (C) Copyright 2018-2021 Texas Instruments, Inc.
 %
 %  Redistribution and use in source and binary forms, with or without
 %  modification, are permitted provided that the following conditions
 %  are met:
 %
 %    Redistributions of source code must retain the above copyright
 %    notice, this list of conditions and the following disclaimer.
 %
 %    Redistributions in binary form must reproduce the above copyright
 %    notice, this list of conditions and the following disclaimer in the
 %    documentation and/or other materials provided with the
 %    distribution.
 %
 %    Neither the name of Texas Instruments Incorporated nor the names of
 %    its contributors may be used to endorse or promote products derived
 %    from this software without specific prior written permission.
 %
 %  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 %  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 %  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 %  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 %  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 %  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 %  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 %  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 %  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 %  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 %  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function High_end_corner_radar_ti_design_visualization(data_port, user_port, loadCfg, max_dist_y, max_dist_x, max_vel,record, filename_rec)

global y_max_dist x_max_dist z_max_dist vel_max;
% Default Values in GUI
y_max_dist = 240;
x_max_dist = 70;
z_max_dist = 20;
vel_max = 60;

user_port = 21; 
data_port = 20;
loadCfg   = 0;
[user_port, data_port, loadCfg, filename_cfg, dimensions, record_options] = choose_ports_dialog(user_port, data_port, loadCfg);

if(isempty(user_port))
    user_port = 54;
end
if(isempty(data_port))
    data_port = 53;
end

% reinitialize COM port connections
if ~isempty(instrfind('Type','serial'))
    disp('Serial port(s) already open. Re-initializing...');
    delete(instrfind('Type','serial'));  % delete open serial ports.
end

if(~record_options.replay && ~(isempty(record_options.filename_rep)))
    errordlg('You have not enabled the checkbox for replaying and have chosen filename only.', 'Invalid Replay Option');
    return;
end

if(record_options.replay && isempty(record_options.filename_rep))
    errordlg('You have enabled the checkbox for replaying but not chosen filename.', 'Invalid Replay Option');
    return;
end

if(~record_options.record && ~(isempty(record_options.filename_rec)))
    errordlg('You have not enabled the checkbox for recording and have chosen filename path only.', 'Invalid Record Option');
    return;
end

if(record_options.record && isempty(record_options.filename_rec))
    errordlg('You have enabled the checkbox for recording but not chosen filename path.', 'Invalid Record Option');
    return;
end

if(~loadCfg && ~isempty(filename_cfg))
    errordlg('You have not enabled the checkbox for loading configuration but chosen filename only.', 'Invalid Load Config. Option');
    return;
end

if(loadCfg && isempty(filename_cfg))
    errordlg('You have enabled the checkbox for loading configuration but not chosen filename only.', 'Invalid Load Config. Option');
    return;
end

if loadCfg
     try
        result = load_config(user_port, filename_cfg);
     catch 
        result = -1;
     end        
else
    if(~(isempty(filename_cfg)))
        errordlg('You have not enabled the checkbox for loading config file.', 'Invalid Config Option');
        return;
    end
    result = 1;
end

if result ~= 1
    if result == -1
       errordlg([{['Could not open port ' num2str(user_port) '. Please note down'], 'the User/Data port from the Device', 'Manager'}]);
    elseif result == -2
       errordlg('Incorrect demo. Make sure you have flashed the High end corner TI Design demo on the AWR2944 Device.', 'Invalid Command');
    end
    return;
end 


if record_options.replay == 0
    read_serial_port_and_plot_object_location(data_port, dimensions, record_options);
else
    read_dat_file_and_plot(record_options.filename_rep, dimensions);
end
end


function out = if_string_convert_to_num(in)

if ischar(in)
    out = str2num(in);
else
    out = in;
end

end

function [user_port, data_port, load_cfg, filename_cfg, dimensions, record_options] = choose_ports_dialog(user_port, data_port, load_cfg)

    global y_max_dist x_max_dist z_max_dist vel_max;
    
    dimensions.max_dist_y = y_max_dist; % meters
    dimensions.max_dist_x = x_max_dist; % meters
    dimensions.max_dist_z = z_max_dist; % meters
    dimensions.max_vel = vel_max; % meters/sec
    
    record_options.replay = 0; 
    record_options.filename_rec = '';
    record_options.filename_rep = '';
    record_options.record = 0; 
    
    load_cfg = 0;
    
    filename_cfg = '';
     
    ba = 0.1;
    of = 0.02;
    dx = (1-ba - of*5)/4;
    d = dialog('Units', 'normalized','Position',[0.1 0.05 0.28 0.92],'Name','High end corner radar TI Design Config.');
    warning off MATLAB:HandleGraphics:ObsoletedProperty:JavaFrame
    jframe=get(d,'javaframe');
    jIcon=javax.swing.ImageIcon('texas_instruments.gif');
    jframe.setFigureIcon(jIcon);

%     h_merge = uipanel('Parent',d,'Title','Grid Mapping options.','FontSize',12, 'Units', 'normalized', 'Position',[.05 (ba + of) .9 dx]);
%     txt_grid_mapping = uicontrol('Parent',h_merge,'Units', 'normalized','Style','text','Position',[0.1 0.7 0.5 0.15],'String',   'Perform Grid Mapping.                       ');
%     txt_filename_rec = uicontrol('Parent',h_merge,'Units', 'normalized','Style','text','Position',[0.1 0.5 0.5 0.15],'String',   'OBD sensor data source file.              ');
%     txt_offset       = uicontrol('Parent',h_merge,'Units', 'normalized','Style','text','Position',[0.1 0.3 0.5 0.15],'String',   'Offset w.r.t sensor (sec).                    ');
%     txt_orientation  = uicontrol('Parent',h_merge,'Units', 'normalized','Style','text','Position',[0.1 0.1 0.5 0.15],'String',   'Radar orientation (deg) w.r.t to car.     ');
%     
%     in_grid_mapping = uicontrol('Parent',h_merge,'Units', 'normalized','Style','checkbox','Position', [0.6 0.7 0.3 0.15],'Value',grid_mapping_options.perfom_grid_mapping,'Callback',@populate_grid_mapping);
%     in_grid_mapping_browse = uicontrol('Parent',h_merge,'Units', 'normalized','Position', [0.65 0.7 0.25 0.15],'String','Browse','Callback',@populate_grid_mapping_filename_picker);
%     in_grid_mapping_file_rec  = uicontrol('Parent',h_merge,'Units', 'normalized','Style','edit','Position',     [0.6 0.5 0.3 0.15],'String',(grid_mapping_options.file),'Callback',@populate_grid_mapping_filename_rec);
%     in_offset        = uicontrol('Parent',h_merge,'Units', 'normalized','Style','edit','Position', [0.6 0.3 0.3 0.15],'String',num2str(grid_mapping_options.offset),'Callback',@populate_grid_mapping_offset);
%     in_orientation   = uicontrol('Parent',h_merge,'Units', 'normalized','Style','edit','Position',     [0.6 0.1 0.3 0.15],'String',num2str(grid_mapping_options.orientation),'Callback',@populate_grid_mapping_orientation);
    
    
    h_record_or_replay = uipanel('Parent',d,'Title','Record & Replay options.','FontSize',12,...
                  'Units', 'normalized', 'Position',[.05 (ba + of) .9 dx]);
    txt_record       = uicontrol('Parent',h_record_or_replay,'Units', 'normalized','Style','text','Position',[0.1 0.7 0.5 0.15],'String',   'Record current session.                     ');
    txt_filename_rec = uicontrol('Parent',h_record_or_replay,'Units', 'normalized','Style','text','Position',[0.1 0.5 0.5 0.15],'String',   'File name to record.                           ');
    txt_replay       = uicontrol('Parent',h_record_or_replay,'Units', 'normalized','Style','text','Position',[0.1 0.3 0.5 0.15],'String',   'Replay pre-existing recording.             ');
    txt_filename_rep = uicontrol('Parent',h_record_or_replay,'Units', 'normalized','Style','text','Position',[0.1 0.1 0.5 0.15],'String',   'File name to replay.                           ');
    
    in_record        = uicontrol('Parent',h_record_or_replay,'Units', 'normalized','Style','checkbox','Position', [0.6 0.7 0.3 0.15],'Value',record_options.replay,'Callback',@populate_record);
    in_filename_rec_browse = uicontrol('Parent',h_record_or_replay,'Units', 'normalized','Position', [0.65 0.7 0.25 0.15],'String','Browse','Callback',@populate_record_filename_picker);
    in_filename_rec  = uicontrol('Parent',h_record_or_replay,'Units', 'normalized','Style','edit','Position',     [0.6 0.5 0.3 0.15],'String',(record_options.filename_rec),'Callback',@populate_filename_rec);
    in_replay        = uicontrol('Parent',h_record_or_replay,'Units', 'normalized','Style','checkbox','Position', [0.6 0.3 0.3 0.15],'Value',record_options.record,'Callback',@populate_replay);
    in_filename_rep_browse = uicontrol('Parent',h_record_or_replay,'Units', 'normalized','Position', [0.65 0.3 0.25 0.15],'String','Browse','Callback',@populate_replay_filename_picker);
    in_filename_rep  = uicontrol('Parent',h_record_or_replay,'Units', 'normalized','Style','edit','Position',     [0.6 0.1 0.3 0.15],'String',(record_options.filename_rep),'Callback',@populate_filename_rep);
    
    
    h_gui_options   = uipanel('Parent',d,'Title',' GUI options.','FontSize',12, 'Units', 'normalized', 'Position',[.05 (ba + dx + 2*of) .9 dx]);          
    
    txt_max_vel     = uicontrol('Parent',h_gui_options,'Units', 'normalized','Style','text','Position',[0.1 0.7 0.5 0.15], 'String',  'Maximum velocity (meters/sec).        ');
    txt_max_range_x = uicontrol('Parent',h_gui_options,'Units', 'normalized','Style','text','Position',[0.1 0.5 0.5 0.15],'String',  'Maximum range width (meters).         ');
    txt_max_range_y = uicontrol('Parent',h_gui_options,'Units', 'normalized','Style','text','Position',[0.1 0.3 0.5 0.15],'String',  'Maximum range depth (meters).         ');
    txt_max_range_z = uicontrol('Parent',h_gui_options,'Units', 'normalized','Style','text','Position',[0.1 0.1 0.5 0.15],'String',  'Maximum height      (meters).         ');
    
    in_max_vel      = uicontrol('Parent',h_gui_options,'Units', 'normalized','Style','edit','Position',[0.6 0.7 0.3 0.15],'String',num2str(dimensions.max_vel),'Callback',@populate_max_vel);
    in_max_range_x  = uicontrol('Parent',h_gui_options,'Units', 'normalized','Style','edit','Position',[0.6 0.5 0.3 0.15],'String',num2str(dimensions.max_dist_x),'Callback',@populate_max_range_x);
    in_max_range_y  = uicontrol('Parent',h_gui_options,'Units', 'normalized','Style','edit','Position',[0.6 0.3 0.3 0.15],'String',num2str(dimensions.max_dist_y),'Callback',@populate_max_range_y);
    in_max_range_z  = uicontrol('Parent',h_gui_options,'Units', 'normalized','Style','edit','Position',[0.6 0.1 0.3 0.15],'String',num2str(dimensions.max_dist_z),'Callback',@populate_max_range_z);
    
    
    h_port_options  = uipanel('Parent',d,'Title',' UART port options.','FontSize',12,'Units', 'normalized','Position',[.05 (ba +3*dx + 4*of) .9 dx]);          
    txt_user        = uicontrol('Parent',h_port_options,'Units', 'normalized','Style','text','Position',[0.1 0.7 0.5 0.15],'String', 'XDS110 Application/User UART');
    txt_data        = uicontrol('Parent',h_port_options,'Units', 'normalized','Style','text','Position',[0.1 0.5 0.5 0.15],'String',  'XDS110 Auxiliary Data Port');
    %txt_reload      = uicontrol('Parent',h_port_options,'Units', 'normalized','Style','text','Position',[0.1 0.3 0.5 0.15],'String',        'Reload configuration.                          ');
    
    in_user         = uicontrol('Parent',h_port_options,'Units', 'normalized','Style','edit','Position',[0.6 0.7 0.3 0.15],'String',num2str(user_port),'Callback',@populate_user_port);
    in_port         = uicontrol('Parent',h_port_options,'Units', 'normalized','Style','edit','Position',[0.6 0.5 0.3 0.15],'String',num2str(data_port),'Callback',@populate_data_port);
    %in_reload       = uicontrol('Parent',h_port_options,'Units', 'normalized','Style','checkbox','Position',[0.6 0.3 0.3 0.15],'Value',load_cfg,'Callback',@populate_reload);
    h_cfg = uipanel('Parent',d,'Title','Config file Options.','FontSize',12, 'Units', 'normalized', 'Position',[.05 (ba +2*dx + 3*of) .9 dx]);
    txt_filename_cfg = uicontrol('Parent',h_cfg,'Units', 'normalized','Style','text','Position',[0.1 0.7 0.5 0.15],'String',   'Load Configuration                        ');
    in_cfg        = uicontrol('Parent',h_cfg,'Units', 'normalized','Style','checkbox','Position', [0.6 0.7 0.3 0.15],'Value',load_cfg,'Callback',@populate_cfg);
    in_cfg_rep_browse = uicontrol('Parent',h_cfg,'Units', 'normalized','Position', [0.65 0.7 0.25 0.15],'String','Browse','Callback',@populate_cfg_filename_picker);
    in_filename_cfg  = uicontrol('Parent',h_cfg,'Units', 'normalized','Style','edit','Position',     [0.6 0.5 0.3 0.15],'String',(filename_cfg),'Callback',@populate_filename_cfg);
    btn = uicontrol('Parent',d,'Units', 'normalized','Position',[0.425 0.02 0.15 0.07],'String','Ok','Callback','delete(gcf)');
       
    % Wait for d to close before running to completion
    uiwait(d);
   
    function populate_user_port(popup,event)
      try 
          user_port = str2num(popup.String); 
      catch
          popup.Value = user_port; 
      end
    end

    function populate_data_port(popup,event)
      try 
          data_port = str2num(popup.String); 
      catch
          popup.Value = data_port; 
      end
    end


    function populate_max_vel(popup,event)
      try 
          dimensions.max_vel = str2num(popup.String); 
      catch
          popup.Value = dimensions.max_vel; 
      end
    end

    function populate_max_range_x(popup,event)
      try 
          dimensions.max_dist_x = str2num(popup.String); 
      catch
          popup.Value = dimensions.max_dist_x; 
      end
    end
    function populate_max_range_y(popup,event)
      try 
          dimensions.max_dist_y = str2num(popup.String); 
      catch
          popup.Value = dimensions.max_dist_y; 
      end
    end

    function populate_max_range_z(popup,event)
      try 
          dimensions.max_dist_z = str2num(popup.String); 
      catch
          popup.Value = dimensions.max_dist_z; 
      end
    end
%     function populate_reload(popup,event)
%       load_cfg = (popup.Value); 
%     end

  
    function populate_replay(popup, event)
        if (record_options.record == 0 && load_cfg == 0) 
            record_options.replay  =  popup.Value; 
        else
            popup.Value = 0; 
        end
    end


    function populate_record(popup, event)
        if record_options.replay  == 0; 
            record_options.record =  popup.Value; 
        else
            popup.Value = 0; 
        end
    end

   function populate_filename_rec(popup,event)
        temp  = (popup.String); 
        if isValidFile(temp)
            record_options.filename_rec = temp;
        else
            popup.String = 'Invalid file';
        end
   end

    function populate_record_filename_picker(popup,event)

        [FileName,PathName] = uiputfile('*.*','Select the filename to record to.');
        if FileName == 0
            return;
        end

        temp = [PathName '\' FileName ];
        record_options.filename_rec = temp;
        in_filename_rec.String = temp;
    end

   function populate_filename_rep(popup,event)
        temp  = (popup.String); 
        if isValidFile(temp) && exist(temp, 'file')
            record_options.filename_rep = temp;
        else
             popup.String = 'NonExistant File';
        end
   end

    function populate_replay_filename_picker(popup,event)
        
        [FileName,PathName] = uigetfile('*.*','Select the filename to replay.');
        if FileName == 0
            return;
        end
        
        temp = [PathName '\' FileName ];
        record_options.filename_rep = temp;
        in_filename_rep.String = temp;
    end

    function populate_cfg(popup, event)
       load_cfg =  popup.Value; 
    end

    function populate_filename_cfg(popup,event)
        temp  = (popup.String); 
        if isValidFile(temp) && exist(temp, 'file')
            filename_cfg = temp;
        else
             popup.String = 'NonExistant File';
        end
   end

    function populate_cfg_filename_picker(popup,event)
        
        [FileName,PathName] = uigetfile('*.*','Select the filename to replay.');
        if FileName == 0
            return;
        end
        
        temp = [PathName '\' FileName ];
        filename_cfg = temp;
        in_filename_cfg.String = temp;
    end

%     function populate_grid_mapping(popup, event)
%         grid_mapping_options.perfom_grid_mapping =  popup.Value; 
%     end
%    
%     function populate_grid_mapping_filename_rec(popup,event)
%         temp  = (popup.String); 
%         if isValidFile(temp) && exist(temp, 'file')
%             grid_mapping_options.file = temp;
%         else
%              popup.String = 'NonExistant File';
%         end
%     end
% 
%     function populate_grid_mapping_filename_picker(popup,event)
%         
%         [FileName,PathName] = uigetfile('*.*','Select the OBD data file');
%         if FileName == 0
%             return;
%         end
%         
%         temp = [PathName '\' FileName ];
%         grid_mapping_options.file = temp;
%         in_grid_mapping_file_rec.String = temp;
%     end
% 
%     function populate_grid_mapping_offset(popup,event)
%       try 
%           grid_mapping_options.offset = str2double(popup.String); 
%       catch
%           popup.Value = grid_mapping_options.offset; 
%       end
%     end
% 
%     function populate_grid_mapping_orientation(popup,event)
%       try 
%           grid_mapping_options.orientation = str2double(popup.String); 
%       catch
%           popup.Value = grid_mapping_options.orientation; 
%       end
%     end
end


function result = isValidFile(filename)

    result = 0; 
    if ~isempty(filename)
    if ~isempty(regexp(filename, '[/\*:?"<>|]', 'once'))
        result = 1;
    else
        
    end
    end
end
