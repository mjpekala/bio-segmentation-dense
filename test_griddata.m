
n = 512;
pts = linspace(1, n, 3);

[xx, yy] = meshgrid(pts, pts);
[xx_omega, yy_omega] = meshgrid(1:512, 1:512);

sigma = 10;
dx = sigma * randn(size(xx));

dx_omega = griddata(xx(:), yy(:), dx(:), xx_omega(:), yy_omega(:));
dx_omega = reshape(dx_omega, n, n);
figure; imagesc(dx_omega);
