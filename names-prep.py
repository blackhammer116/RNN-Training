# A list of names including English and Spanish names
names = [
    "Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy",
    "Adam", "Bella", "Caleb", "Diana", "Ethan", "Fiona", "George", "Hannah", "Isaac", "Jasmine",
    "Kyle", "Lily", "Mason", "Nora", "Owen", "Penelope", "Quentin", "Ruby", "Samuel", "Sophia",
    "Thomas", "Ursula", "Victor", "Willow", "Xavier", "Yara", "Zachary", "Aria", "Brandon", "Chloe",
    "Dylan", "Ella", "Finn", "Gabriella", "Henry", "Isabelle", "Jack", "Katherine", "Leo", "Mia",
    "Nathan", "Olivia", "Patrick", "Quinn", "Ryan", "Scarlett", "Tyler", "Victoria", "William", "Zoe",
    "Alexander", "Aurora", "Benjamin", "Brooklyn", "Carter", "Claire", "Daniel", "Eleanor", "Elijah", "Elizabeth",
    "Ethan", "Emily", "Felix", "Florence", "Gabriel", "Georgia", "Harrison", "Hazel", "Isaac", "Isla",
    "Jasper", "Josephine", "Julian", "Layla", "Lucas", "Luna", "Matthew", "Maya", "Michael", "Mila",
    "Noah", "Natalie", "Oliver", "Nora", "Sebastian", "Stella", "Theodore", "Sophie", "Vincent", "Violet",
    "Wyatt", "Zara", "Adrian", "Amelia", "Caleb", "Charlotte", "Dominic", "Eleanor", "Everett", "Eva",
    "Frederick", "Freya", "Grayson", "Genevieve", "Hugo", "Harper", "Ian", "Ivy", "Jonathan", "Julia",
    "Kai", "Kinsley", "Liam", "Lucy", "Miles", "Nova", "Oscar", "Piper", "Rhys", "Rose",
    "Silas", "Skylar", "Tristan", "Taylor", "Uriah", "Uma", "Wesley", "Willa", "Xander", "Ximena",
    "Yousef", "Yasmine", "Zane", "Zaylee", "Aaron", "Abigail", "Austin", "Aubrey", "Cameron", "Caroline",
    "Declan", "Delilah", "Ezra", "Elena", "Graham", "Gianna", "Hunter", "Hailey", "Isaiah", "Iris",
    "Jordan", "Jocelyn", "Kevin", "Kennedy", "Logan", "Lydia", "Marcus", "Madelyn", "Nicholas", "Naomi",
    "Parker", "Paisley", "Rowan", "Riley", "Sawyer", "Sarah", "Tristan", "Summer", "Ulysses", "Victoria",
    "Walker", "Whitney", "York", "Yvette", "Zion", "Zendaya",
    "Sofía", "Santiago", "Isabella", "Mateo", "Valeria", "Diego", "Camila", "Sebastián", "María", "Alejandro",
    "Lucía", "Benjamín", "Emma", "Gabriel", "Mariana", "Nicolás", "Victoria", "Daniel", "Paula", "David",
    "Elena", "Javier", "Gabriela", "Manuel", "Antonella", "Miguel", "Carolina", "Adrián", "Andrea", "Carlos",
    "Daniela", "Andrés", "Natalia", "Fernando", "Valentina", "José", "Regina", "Jorge", "Emilia", "Ricardo",
    "Catalina", "Sergio", "Isabela", "Pedro", "Renata", "Pablo", "Jimena", "Luis", "Alexa", "Hugo",
    "Alexandra", "Ángel", "Fernanda", "Rafael", "Victoria", "Eduardo", "Martina", "Christian", "Ana Sofía", "Marco",
    "Guadalupe", "Gonzalo", "María José", "Cristian", "Samantha", "Leonardo", "María Fernanda", "Mauricio", "Alejandra", "Emilio",
    "Ximena", "Esteban", "Romina", "Álvaro", "Juana", "Roberto", "Valery", "Joaquín", "Sara", "Emmanuel",
    "Hanna", "Eric", "Zoe", "Francisco", "Paula Sofía", "Jesús", "Arantza", "Patricio", "Ashley", "Gustavo",
    "Montserrat", "Alfredo", "Rebecca", "Enrique", "Ashley", "Miguel Ángel", "Melanie", "Arturo", "María Camila", "Federico",
    "Victoria Sofía", "Salvador", "Nicole", "Diego Alejandro", "Silvana", "Ricardo Andrés", "María Paula", "Juan José", "Valeria Sofía", "Juan Pablo",
    "Gabriela Alejandra", "Santiago José", "Andrea Carolina", "Carlos Andrés", "Daniela Alejandra", "Andrés Felipe", "Natalia Andrea", "Fernando José", "Valentina Andrea", "José Antonio",
    "Regina Sofía", "Jorge Andrés", "Emilia Sofía", "Ricardo José", "Catalina Andrea", "Sergio Andrés", "Isabela Sofía", "Pedro Antonio", "Renata Sofía", "Pablo Andrés",
    "Jimena Alejandra", "Luis Fernando", "Alexa Sofía", "Hugo Andrés", "Alexandra María", "Ángel David", "Fernanda Sofía", "Rafael Antonio", "Victoria Alejandra", "Eduardo José",
    "Martina Alejandra", "Christian Andrés", "Ana Sofía María", "Marco Antonio", "Guadalupe Sofía", "Gonzalo Andrés", "María José Alejandra", "Cristian David", "Samantha Nicole", "Leonardo Andrés",
    "María Fernanda Sofía", "Mauricio José", "Alejandra Sofía", "Emilio José", "Ximena Alejandra María", "Esteban Andrés", "Romina Sofía", "Álvaro José", "Juana María", "Roberto Andrés",
    "Valery Sofía", "Joaquín Andrés", "Sara Alejandra", "Emmanuel José", "Hanna Sofía", "Eric Andrés", "Zoe Alejandra", "Francisco José", "Paula Sofía María", "Jesús Alberto",
    "Arantza Sofía", "Patricio José", "Ashley Sofía", "Gustavo Adolfo José", "Montserrat Sofía", "Alfredo José Andrés", "Rebecca Sofía", "Enrique José Andrés", "Ashley Nicole Alejandra", "Miguel Ángel José Andrés",
    "Melanie Sofía", "Arturo José Andrés", "María Camila Sofía Alejandra", "Federico José Andrés", "Victoria Sofía Alejandra María", "Salvador Andrés José", "Nicole Alejandra Sofía", "Diego Alejandro José Andrés", "Silvana Sofía Alejandra", "Ricardo Andrés José Antonio",
    "María Paula Sofía Alejandra", "Juan José Andrés Felipe", "Valeria Sofía Alejandra María", "Juan Pablo Andrés José", "Gabriela Alejandra María José", "Santiago José Andrés Felipe", "Andrea Carolina Alejandra", "Carlos Andrés José", "Daniela Alejandra María", "Andrés Felipe José",
    "Natalia Andrea Alejandra", "Fernando José Andrés", "Valentina Andrea Alejandra", "José Antonio Andrés", "Regina Sofía Alejandra", "Jorge Andrés José", "Emilia Sofía Alejandra", "Ricardo José Andrés", "Catalina Andrea Alejandra", "Sergio Andrés José",
    "Isabela Sofía Alejandra", "Pedro Antonio Andrés", "Renata Sofía Alejandra", "Pablo Andrés José", "Jimena Alejandra María", "Luis Fernando José", "Alexa Sofía Alejandra", "Hugo Andrés José", "Alexandra María Alejandra", "Ángel David José",
    "Fernanda Sofía Alejandra", "Rafael Antonio José", "Victoria Alejandra María", "Eduardo José Andrés", "Martina Alejandra María", "Christian Andrés José", "Ana Sofía María Alejandra", "Marco Antonio José", "Guadalupe Sofía Alejandra", "Gonzalo Andrés José",
    "María José Alejandra María", "Cristian David José", "Samantha Nicole Alejandra", "Leonardo Andrés José", "María Fernanda Sofía Alejandra", "Mauricio José Andrés", "Alejandra Sofía Alejandra", "Emilio José Andrés", "Ximena Alejandra María", "Esteban Andrés José",
    "Romina Sofía Alejandra", "Álvaro José Andrés", "Juana María Alejandra", "Roberto Andrés José", "Valery Sofía Alejandra", "Joaquín Andrés José", "Sara Alejandra María", "Emmanuel José Andrés", "Hanna Sofía Alejandra", "Eric Andrés José",
    "Zoe Alejandra María", "Francisco José Andrés", "Paula Sofía María Alejandra", "Jesús Alberto José", "Arantza Sofía Alejandra", "Patricio José Andrés", "Ashley Sofía Alejandra", "Gustavo Adolfo José", "Montserrat Sofía Alejandra", "Alfredo José Andrés",
    "Rebecca Sofía Alejandra", "Enrique José Andrés", "Ashley Nicole Alejandra", "Miguel Ángel José Andrés", "Melanie Sofía Alejandra", "Arturo José Andrés", "María Camila Sofía Alejandra", "Federico José Andrés", "Victoria Sofía Alejandra María", "Salvador Andrés José",
    "Nicole Alejandra Sofía", "Diego Alejandro José Andrés", "Silvana Sofía Alejandra", "Ricardo Andrés José Antonio", "María Paula Sofía Alejandra", "Juan José Andrés Felipe", "Valeria Sofía Alejandra María", "Juan Pablo Andrés José", "Gabriela Alejandra María José", "Santiago José Andrés Felipe"
]


# Specify the filename
filename = "names.txt"

# Write the names to the file, one name per line
with open(filename, "w") as f:
    for name in names:
        f.write(name + "\\n")

print(f"Names have been saved to {filename}")