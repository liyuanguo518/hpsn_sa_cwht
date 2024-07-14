% 读取第一个Excel文件中的数据
filename1 = 'mah.xlsx';
data_table1 = readtable(filename1, 'ReadVariableNames', false); % 不读取变量名
data_array1 = table2array(data_table1);

% 读取第二个Excel文件中的数据
filename2 = 'euc.xlsx';
data_table2 = readtable(filename2, 'ReadVariableNames', false); % 不读取变量名
data_array2 = table2array(data_table2);

% 选择特定的特征值编号作为x轴数据（2到10）
x = 2:10;

% 计算每个数据集的平均值和标准差
mean_vals1 = mean(data_array1, 1);
std_vals1 = std(data_array1, 0, 1);

mean_vals2 = mean(data_array2, 1);
std_vals2 = std(data_array2, 0, 1);

% 创建一个新的图形窗口
figure;

% 绘制折线图并标出误差棒（数据集1）
h1 = errorbar(x, mean_vals1, std_vals1, '-o', 'LineWidth', 2, ...
    'MarkerSize', 6, 'CapSize', 5); % 设置误差棒帽子的大小为5
hold on;

% 绘制折线图并标出误差棒（数据集2）
h2 = errorbar(x, mean_vals2, std_vals2, '-s', 'LineWidth', 2, ...
    'MarkerSize', 6, 'CapSize', 5); % 设置误差棒帽子的大小为5

% 添加图例
legend([h1, h2], '数据集1', '数据集2', 'Location', 'best');

% 添加标题和轴标签
title('两个数据集的平均值及其标准差');
xlabel('特征值编号');
ylabel('值');

% 设置y轴范围
ylim([0.85, 0.94]);

% 优化图形显示
grid on; % 添加网格线
hold off;