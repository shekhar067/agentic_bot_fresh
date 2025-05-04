window.openTab = function(evt, tabName) {
    const tabContents = document.querySelectorAll('.tab-content');
    const tabButtons = document.querySelectorAll('.tab-buttons .tab-button');
  
    tabContents.forEach(tc => {
      tc.classList.remove('active');
      tc.style.display = 'none';
    });
    tabButtons.forEach(btn => btn.classList.remove('active'));
  
    const activeTab = document.getElementById(tabName);
    if (activeTab) {
      activeTab.classList.add('active');
      activeTab.style.display = 'block';
    }
    if (evt && evt.currentTarget) {
      evt.currentTarget.classList.add('active');
    }
  };

  function openTab(evt, tabName) {
    const tabContents = document.querySelectorAll('.tab-content');
    tabContents.forEach(tc => tc.classList.remove('active'));

    const tabButtons = document.querySelectorAll('.tab-buttons button');
    tabButtons.forEach(tb => tb.classList.remove('active'));

    const targetTab = document.getElementById(tabName);
    if (targetTab) targetTab.classList.add('active');
    if (evt && evt.currentTarget) evt.currentTarget.classList.add('active');
}

window.addEventListener('DOMContentLoaded', () => {
    const defaultBtn = document.querySelector('.tab-buttons .tab-button');
    if (defaultBtn) defaultBtn.click();  // Trigger default tab open
});
