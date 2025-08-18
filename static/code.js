
function searchMembers(this_context) {
    const searchInput = document.getElementById("member-search");
    const filter = searchInput.value.toLowerCase();
    const members = document.getElementsByClassName("member");

    Array.from(members).forEach(member => { // Convert HTMLCollection to Array
        // get employee id from member
        let employeeId = member.getAttribute("data-employee-id");
        let txt = member.getAttribute("data-employee-text");
        if (txt && txt.includes(filter)) {
            member.style.display = "";
        } else {
            member.style.display = "none";
        }
    });

    // save searchInput to localStorage
    localStorage.setItem("memberSearch", searchInput.value);
}

function startup() {
    // Load search input from localStorage
    const searchInput = document.getElementById("member-search");
    const savedSearch = localStorage.getItem("memberSearch");
    if (savedSearch) {
        searchInput.value = savedSearch;
        searchMembers(); // Call the function to filter members based on saved search
    }

    // Add event listener for search input
    // searchInput.addEventListener("input", searchMembers);
    searchInput.addEventListener("keyup", searchMembers);
}

startup();


