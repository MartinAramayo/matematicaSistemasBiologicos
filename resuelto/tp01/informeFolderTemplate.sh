#!/bin/bash

# Hacer mas generales

# esto va un directorio arriba del lugar al cual vas a crear todo

echo "Con este script podes crear una familia de carpetas
para organizar los archivos del informe. Ingresa el codename
del informe. Tiene que ser minuscula con una
sola palabra, sin espacios porfis "

# Lee de entrada
read -p 'Ingrese nombre en clave: ' name

# Crea carpetas para guardar lo necesario para un informe
mkdir $name
mkdir $name/introduccion
mkdir $name/metdExp
mkdir $name/resultados
mkdir $name/discusion
mkdir $name/conclusiones
mkdir $name/correccion
mkdir $name/bibliografia

# Template md de informe
touch $name/$name.md

# Contenido del template
echo "#Informe

# Metodo experimental

# Resultados

# Discusión

# Conclusiones

" > $name/$name.md

# Esto es el template de las correciones

touch "$name/correciones("$name").md"

echo "# Correciones

# Posibles preguntas

# Cuentas

# Dimensiones a memorizar

# Posibles mejoras a la metodología

# Explicaciones posibles

# Que puedo concluir

" > "$name/correciones("$name").md"

#estos son comentarios
: ' esto es un comentario multilinea.
si multilinea,
como puedes ver
'

cp main.tex $name
