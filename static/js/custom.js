// JavaScript for smooth scroll
document.querySelectorAll('.smoothscroll').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const targetId = this.getAttribute('href').substring(1);
        document.getElementById(targetId).scrollIntoView({ behavior: 'smooth' });
    });
});

// Example for form validation (Symptom page)
document.querySelector('form').addEventListener('submit', function (e) {
    const age = document.querySelector('input[name="age"]').value;
    if (age < 0 || age > 120) {
        alert('Please enter a valid age between 0 and 120');
        e.preventDefault();
    }
});
