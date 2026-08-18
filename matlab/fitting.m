%Programa para hacer fitting con Matlab o GNU Octane
clear all
clc
datos_ttc = importdata("../data/B1654run34.dat");

%% Bucle para leer datos
texto = sprintf(datos_ttc.textdata{2,1}); %Definicion de variables y datos
%texto = sprintf(datos_ttc{2,1}) %Usar si es data B2356 o B1965r
var = strsplit(texto,"\t");
data = datos_ttc.data; %Data para runs que no sean B2356


if length(var) ~= size(data,2) %Comp de que sea possible hacerlo
    error('El número de encabezados (%d) no coincide con el número de columnas de data (%d)', ...
        length(var), size(data,2));
end

s = struct(); %Bucle de funcionamento
for cont = 1:length(var)-1
    nombre = genvarname(var{cont});
    s.(nombre) = data(:,cont);
end

clear ("texto","nombre","var","cont","data") %limpiar variables que no queremos

%% Graficar

subplot(2,2,1)
plot3(s.SA, s.FY, s.FZ, ".")
xlabel("SA"); ylabel("FY"); zlabel("FZ")
title("SA vs FY vs FZ")
grid on

subplot(2,2,2)
plot3(s.IA, s.FX, s.FZ, '.')
xlabel('IA'); ylabel('FX'); zlabel('FZ')
title('IA vs FX vs FZ')
grid on

subplot(2,2,3)
plot3(s.RL, s.N, s.FZ, '.')
xlabel('RL'); ylabel('N'); zlabel('FZ')
title('RL vs N vs FZ')
grid on

subplot(2,2,4)
plot3(s.SA, s.FX, s.MZ, '.')
xlabel('SA'); ylabel('FY'); zlabel('MZ')
title('SA vs FY vs MZ')
grid on
