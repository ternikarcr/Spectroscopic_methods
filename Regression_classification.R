#install.packages("soiltexture")
#install.packages("caret")
library(readxl)
library(writexl)
library(soiltexture)
#library(caret)

mapping <- c("Sa" = 1, "LoSa" = 2, "SaLo" = 3, "SaClLo" = 4, "SaCl" = 5, "Cl" = 6, "ClLo" = 7, "Lo" = 8, "SiCl" = 9, "SiClLo" = 10, "SiLo" = 11, "Si" = 12)
         
#M1 and M2 (Use respective lines of codes)
#wd = "~/Documents/HSI/Soil/Paper_2_Quantitative/M1"
#wd = "~/Documents/HSI/Soil/Paper_2_Quantitative/M2"
wd = "~/Documents/HSI/Soil/Paper_2_Quantitative/M3"

for (i in 1:100) {
  #Pred
  #foo <- read.csv(paste0("iter_p_",i))
  foo <- read.csv(paste0("iter_p_log_ratio_",i))
  colnames(foo) = c("CLAY", "SILT", "SAND")
  foo = as.data.frame(foo)
  IP_norm = TT.normalise.sum(foo)
  a = TT.points.in.classes(tri.data = IP_norm, class.sys = "USDA.TT", PiC.type = "t", collapse = ";")
  #Actual
  rm(foo,IP_norm)
  foo <- read.csv(paste0("iter_",i))
  colnames(foo) = c("CLAY", "SILT", "SAND")
  foo = as.data.frame(foo)
  IP_norm = TT.normalise.sum(foo)
  b = TT.points.in.classes(tri.data = IP_norm, class.sys = "USDA.TT", PiC.type = "t", collapse = ";")
  #Converting to texture and saving
  data_texture = as.data.frame(foo)
  rm(foo,IP_norm)
  data_texture$actual = mapping[b]
  data_texture$pred = mapping[a]
  data_texture = data_texture[,-c(1:3)]
  write.csv(data_texture, file = paste0(wd,"/iter_tt_",i,".csv"), quote = TRUE, eol = "\n", na = "-9999", row.names = FALSE)
}

#Revision
wd = "C:/Refined/Draft_2/Working_files/Revision"
setwd(wd)

mapping <- c("Sa" = 1, "LoSa" = 2, "SaLo" = 3, "SaClLo" = 4, "SaCl" = 5, "Cl" = 6, "ClLo" = 7, "Lo" = 8, "SiCl" = 9, "SiClLo" = 10, "SiLo" = 11, "Si" = 12)
clay <- read.csv("Y_p_iter_pls2_clay.csv")
silt <- read.csv("Y_p_iter_pls2_silt.csv")
sand <- read.csv("Y_p_iter_pls2_sand.csv")
clay_act <- read.csv("Y_iter_pls2_clay.csv")
silt_act <- read.csv("Y_iter_pls2_silt.csv")
sand_act <- read.csv("Y_iter_pls2_sand.csv")

for (i in 1:100) {
  foo = NULL
  foo$CLAY = clay[,i]
  foo$SILT = silt[,i]
  foo$SAND = sand[,i]
  foo = as.data.frame(foo)
  IP_norm = TT.normalise.sum(foo)
  a = TT.points.in.classes(tri.data = IP_norm, class.sys = "USDA.TT", PiC.type = "t", collapse = ";")
  #Actual
  rm(foo,IP_norm)
  foo = NULL
  foo$CLAY = clay_act[,i]
  foo$SILT = silt_act[,i]
  foo$SAND = sand_act[,i]
  foo = as.data.frame(foo)
  IP_norm = TT.normalise.sum(foo)
  b = TT.points.in.classes(tri.data = IP_norm, class.sys = "USDA.TT", PiC.type = "t", collapse = ";")
  #Converting to texture and saving
  data_texture = as.data.frame(foo)
  rm(foo,IP_norm)
  data_texture$actual = mapping[b]
  data_texture$pred = mapping[a]
  data_texture = data_texture[,-c(1:3)]
  write.csv(data_texture, file = paste0(wd,"/iter_tt_",i,".csv"), quote = TRUE, eol = "\n", na = "-9999", row.names = FALSE)
}
