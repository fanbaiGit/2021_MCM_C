%% 绘制华盛顿州的大致区域
rectangle('Position', [0 0 547 386],'EdgeColor','r','LineWidth',2);
grid on;
hold on;
set(gca,'xaxislocation','top','yaxislocation','left','ydir','reverse')

%% 绘制2020年出现亚洲大黄蜂的散点图
x = [153.50937761465937, 155.66533604736958, 180.4658077064662, 144.17133886997448, 163.0974937493654, 162.3718356580837, 162.24618539097497, 163.10739009783708, 163.10672292827732];
y = [6.195002948147994, 5.439099836818695, 25.99359322199144, 9.316022149207983, 3.005710062129169, 3.5363322520765093, 3.10511832654888, 3.0164959700133225, 3.0111586135344277];
sz = 25;
c = linspace(1,10,length(x));
scatter(x,y,sz,c,'filled')

%% 预测2021年亚洲大黄蜂的分布
mu = [9.5 3.7];
Sigma = [70.56 40.685568; 40.685568 49.5616];
rng('default')  %对于重现性

k = 100; %亚洲大黄蜂每年的繁殖量

n = 1000000; %随机取点数
width_min = 40;
width_max = 120;
height_min = 40;
height_max = 120;
X = randi([width_min,width_max],1,n); %在地图上随机取点
Y = randi([height_min,height_max],1,n);
s = 0; %地区亚洲大黄蜂数量
S = zeros(1,n);
N = 0;
for j=1:n
    for i=1:9
        p = mvnpdf([X(j)-x(i),Y(j)-y(i)],mu,Sigma);
        s = s + 100 * k * p;
    end
        if s >= 1
            S(j) = s;
            N = N + 1;
        end
end
