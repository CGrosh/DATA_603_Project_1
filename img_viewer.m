data = load('data.mat');
face = data.face;
face_net = face(:,:,1:3:end);
face_exp = face(:,:,2:3:end);
face_illum = face(:,:,3:3:end);

check_net = face(:,:,3*1-2);
check_exp = face(:,:,3*1-1);
check_illum = face(:,:,3*1);

first_net = face_net(:,:,1);
figure;
imshow(check_net);
