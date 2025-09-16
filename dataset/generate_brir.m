function generate_brir

room.name = 'Laboratory';

% Room dimensions [x, y, z] in meters:
room.boxsize = [4, 5, 3];

room.materials = {...
    'hall:plaster_sprayed'; 'hall:plaster_sprayed'; 'hall:plaster_sprayed'; ...     % -z, -y, -x
    'hall:plaster_sprayed'; 'hall:plaster_sprayed'; 'hall:plaster_sprayed'}; % +x, +y, +z

op.spat_mode = 'hrtf';
op.hrtf_database = 'cipic.sofa';

room.freq = [250 500 1e3 2e3 4e3];

for mic_num = [21, 41, 61, 81, 101, 121, 141, 161, 181, 201, 221]
    ls_y = linspace(0.5, 4.5, mic_num);
    ls_src = linspace(2, 3, 6);
    brir = zeros(1000, mic_num, 6, 2);
    
    for i = 1:length(ls_src)
        room.srcpos = [3, ls_src(i), 1];
        for j = 1:length(ls_y)
            room.recpos = [2, ls_y(j), 1];
            room.recdir	= [0, 0];
            op.ism_only = 1;
            ir = razr(room, op).sig;
            brir(:, j, i, :) = ir(1:1000, :);
            disp(i);
            disp(j);
        end
    end
    
    save("brir_" + mic_num + ".mat", "brir")

end

end