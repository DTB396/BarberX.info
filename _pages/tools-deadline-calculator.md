---
layout: page
title: Deadline Calculator
subtitle: Court deadline calculator for NJ and Federal courts
description: Calculate filing deadlines based on FRCP, NJ court rules, and local rules
permalink: /tools/deadline-calculator/
toc: false
---

<div class="tool-page-header">
  <div class="tool-page-header__icon">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="48" height="48">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>
      <line x1="16" y1="2" x2="16" y2="6"/>
      <line x1="8" y1="2" x2="8" y2="6"/>
      <line x1="3" y1="10" x2="21" y2="10"/>
    </svg>
  </div>
  <h1>Deadline Calculator</h1>
  <p>Calculate accurate court deadlines with automatic holiday and weekend adjustments</p>
</div>

<div class="calculator-container">
  <div class="calculator-form">
    <div class="form-group">
      <label for="courtSystem">Court System</label>
      <select id="courtSystem" class="form-select">
        <option value="nj-superior">NJ Superior Court</option>
        <option value="federal">Federal Court (FRCP)</option>
        <option value="nj-municipal">NJ Municipal Court</option>
      </select>
    </div>

    <div class="form-group">
      <label for="deadlineType">Deadline Type</label>
      <select id="deadlineType" class="form-select">
        <option value="answer">Answer to Complaint</option>
        <option value="motion-oppose">Opposition to Motion</option>
        <option value="motion-reply">Reply Brief</option>
        <option value="discovery">Discovery Response</option>
        <option value="appeal">Notice of Appeal</option>
        <option value="custom">Custom Days</option>
      </select>
    </div>

    <div class="form-group" id="customDaysGroup" style="display:none;">
      <label for="customDays">Number of Days</label>
      <input type="number" id="customDays" class="form-input" min="1" value="30" />
    </div>

    <div class="form-group">
      <label for="triggerDate">Trigger Date</label>
      <input type="date" id="triggerDate" class="form-input" />
    </div>

    <div class="form-group">
      <label for="serviceMethod">Service Method</label>
      <select id="serviceMethod" class="form-select">
        <option value="hand">Hand Delivery</option>
        <option value="mail">Mail</option>
        <option value="electronic">Electronic (E-Filed)</option>
      </select>
    </div>

    <div class="form-checkbox">
      <label>
        <input type="checkbox" id="excludeHolidays" checked />
        <span>Exclude Holidays & Weekends</span>
      </label>
    </div>

    <button class="btn-premium btn-premium--large" id="calculateBtn">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
        <path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"/>
      </svg>
      Calculate Deadline
    </button>

  </div>

  <div class="calculator-result" id="result" style="display:none;">
    <div class="result-header">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="32" height="32">
        <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"/>
      </svg>
      <h3>Deadline Calculated</h3>
    </div>

    <div class="result-deadline">
      <span class="result-label">Filing Deadline:</span>
      <span class="result-date" id="deadlineDate">—</span>
    </div>

    <div class="result-details">
      <div class="detail-item">
        <span class="detail-label">Calendar Days:</span>
        <span class="detail-value" id="calendarDays">—</span>
      </div>
      <div class="detail-item">
        <span class="detail-label">Business Days:</span>
        <span class="detail-value" id="businessDays">—</span>
      </div>
      <div class="detail-item">
        <span class="detail-label">Holidays Excluded:</span>
        <span class="detail-value" id="holidaysExcluded">—</span>
      </div>
    </div>

    <div class="result-warning">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
        <circle cx="12" cy="12" r="10"/>
        <line x1="12" y1="8" x2="12" y2="12"/>
        <line x1="12" y1="16" x2="12.01" y2="16"/>
      </svg>
      <p>Always verify deadlines with court clerk or local rules. This calculator is for informational purposes only.</p>
    </div>

    <div class="result-actions">
      <button class="btn-secondary" id="resetBtn">Calculate Another</button>
      <button class="btn-premium">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="18" height="18">
          <path d="M21 15v4a2 2 0 01-2 2H5a2 2 0 01-2-2v-4M7 10l5 5 5-5M12 15V3"/>
        </svg>
        Export to Calendar
      </button>
    </div>

  </div>
</div>

<div class="info-cards">
  <div class="info-card">
    <h3>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
        <path d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253"/>
      </svg>
      Common NJ Deadlines
    </h3>
    <ul>
      <li><strong>Answer to Complaint:</strong> 35 days after service</li>
      <li><strong>Motion Opposition:</strong> 16 days before return date</li>
      <li><strong>Reply Brief:</strong> 2 days before return date</li>
      <li><strong>Discovery Response:</strong> 30 days from service</li>
    </ul>
  </div>

  <div class="info-card">
    <h3>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
        <circle cx="12" cy="12" r="10"/>
        <path d="M12 16v-4M12 8h.01"/>
      </svg>
      Important Notes
    </h3>
    <ul>
      <li>E-filed documents due by 11:59 PM</li>
      <li>Mail adds 3 days for service (FRCP 6(d))</li>
      <li>Holidays extend deadlines to next business day</li>
      <li>Always check local court rules</li>
    </ul>
  </div>

  <div class="info-card info-card--premium">
    <h3>
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20">
        <path d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"/>
      </svg>
      Premium Features
    </h3>
    <ul>
      <li>Automatic calendar sync (Google, Outlook)</li>
      <li>Email deadline reminders</li>
      <li>Track multiple cases</li>
      <li>Custom court rule templates</li>
    </ul>
    <a href="/contact/" class="btn-premium btn-premium--small">Upgrade Now</a>
  </div>
</div>

<style>
.calculator-container {
  max-width: 800px;
  margin: 2rem auto;
}

.calculator-form {
  background: linear-gradient(135deg, rgb(255 255 255 / 8%) 0%, rgb(255 255 255 / 3%) 100%);
  border: 1px solid rgb(255 255 255 / 12%);
  border-radius: 20px;
  padding: 2rem;
}

.form-group {
  margin-bottom: 1.5rem;
}

.form-group label {
  display: block;
  font-size: 0.95rem;
  font-weight: 500;
  color: rgb(255 255 255 / 80%);
  margin-bottom: 0.5rem;
}

.form-select,
.form-input {
  width: 100%;
  padding: 0.875rem 1rem;
  background: rgb(255 255 255 / 5%);
  border: 1px solid rgb(255 255 255 / 12%);
  border-radius: 12px;
  color: #f5f5f7;
  font-size: 1rem;
  outline: none;
  transition: all 0.3s ease;
}

.form-select:focus,
.form-input:focus {
  border-color: #d4a574;
  box-shadow: 0 0 0 3px rgb(212 165 116 / 20%);
}

.form-checkbox {
  margin: 1.5rem 0;
}

.form-checkbox label {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  cursor: pointer;
  user-select: none;
}

.form-checkbox input[type="checkbox"] {
  width: 20px;
  height: 20px;
  cursor: pointer;
}

.form-checkbox span {
  font-size: 0.95rem;
  color: rgb(255 255 255 / 70%);
}

.calculator-result {
  background: linear-gradient(135deg, rgb(212 165 116 / 10%) 0%, rgb(212 165 116 / 3%) 100%);
  border: 2px solid rgb(212 165 116 / 30%);
  border-radius: 20px;
  padding: 2rem;
  margin-top: 2rem;
}

.result-header {
  display: flex;
  align-items: center;
  gap: 1rem;
  margin-bottom: 2rem;
  color: #d4a574;
}

.result-header h3 {
  font-size: 1.5rem;
  color: #f5f5f7;
  margin: 0;
}

.result-deadline {
  text-align: center;
  padding: 2rem;
  background: rgb(255 255 255 / 5%);
  border-radius: 16px;
  margin-bottom: 2rem;
}

.result-label {
  display: block;
  font-size: 0.9rem;
  color: rgb(255 255 255 / 60%);
  margin-bottom: 0.5rem;
}

.result-date {
  display: block;
  font-size: 2.25rem;
  font-weight: 700;
  background: linear-gradient(135deg, #d4a574 0%, #e8c9a8 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}

.result-details {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 1rem;
  margin-bottom: 2rem;
}

.detail-item {
  padding: 1rem;
  background: rgb(255 255 255 / 3%);
  border: 1px solid rgb(255 255 255 / 8%);
  border-radius: 12px;
  text-align: center;
}

.detail-label {
  display: block;
  font-size: 0.85rem;
  color: rgb(255 255 255 / 50%);
  margin-bottom: 0.5rem;
}

.detail-value {
  display: block;
  font-size: 1.5rem;
  font-weight: 600;
  color: #f5f5f7;
}

.result-warning {
  display: flex;
  gap: 1rem;
  padding: 1rem 1.25rem;
  background: rgb(245 158 11 / 10%);
  border: 1px solid rgb(245 158 11 / 25%);
  border-radius: 12px;
  margin-bottom: 2rem;
  color: rgb(251 191 36);
}

.result-warning svg {
  flex-shrink: 0;
}

.result-warning p {
  font-size: 0.875rem;
  margin: 0;
}

.result-actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
}

.info-cards {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  margin: 3rem 0;
}
</style>

<script>
document.addEventListener('DOMContentLoaded', function() {
  const deadlineTypeSelect = document.getElementById('deadlineType');
  const customDaysGroup = document.getElementById('customDaysGroup');
  const triggerDateInput = document.getElementById('triggerDate');
  const calculateBtn = document.getElementById('calculateBtn');
  const resultDiv = document.getElementById('result');
  const resetBtn = document.getElementById('resetBtn');

  // Set today as default trigger date
  const today = new Date().toISOString().split('T')[0];
  triggerDateInput.value = today;

  deadlineTypeSelect.addEventListener('change', function() {
    customDaysGroup.style.display = this.value === 'custom' ? 'block' : 'none';
  });

  calculateBtn.addEventListener('click', calculateDeadline);
  resetBtn.addEventListener('click', () => {
    resultDiv.style.display = 'none';
    window.scrollTo({ top: 0, behavior: 'smooth' });
  });

  function calculateDeadline() {
    const deadlineType = document.getElementById('deadlineType').value;
    const triggerDate = new Date(document.getElementById('triggerDate').value);
    const serviceMethod = document.getElementById('serviceMethod').value;
    const excludeHolidays = document.getElementById('excludeHolidays').checked;

    let days = 30;
    switch(deadlineType) {
      case 'answer': days = 35; break;
      case 'motion-oppose': days = 16; break;
      case 'motion-reply': days = 2; break;
      case 'discovery': days = 30; break;
      case 'appeal': days = 45; break;
      case 'custom': days = parseInt(document.getElementById('customDays').value); break;
    }

    // Add service method extension
    if (serviceMethod === 'mail') days += 3;

    const deadline = new Date(triggerDate);
    deadline.setDate(deadline.getDate() + days);

    // Simple weekend adjustment
    while (excludeHolidays && (deadline.getDay() === 0 || deadline.getDay() === 6)) {
      deadline.setDate(deadline.getDate() + 1);
    }

    // Display results
    document.getElementById('deadlineDate').textContent = deadline.toLocaleDateString('en-US', {
      weekday: 'long',
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
    document.getElementById('calendarDays').textContent = days;
    document.getElementById('businessDays').textContent = Math.floor(days * 5/7);
    document.getElementById('holidaysExcluded').textContent = excludeHolidays ? 'Yes' : 'No';

    resultDiv.style.display = 'block';
    resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
});
</script>
