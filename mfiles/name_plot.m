gcf;
titlestr = input("title: ", "s");
xlabelstr = input("xlabel: ", "s");
ylabelstr = input("ylabel: ", "s");

plt = title(titlestr);
plt.FontSize = 14;
plt.FontName = "DejaVu Sans";
plt.FontAngle = "normal";
plt.FontWeight = "normal";

plt = xlabel(xlabelstr);
plt.FontSize = 12;
plt.FontName = "DejaVu Sans";
plt.FontAngle = "italic";

plt = ylabel(ylabelstr);
plt.FontSize = 12;
plt.FontName = "DejaVu Sans";
plt.FontAngle = "italic";

grid on;