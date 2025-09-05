document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('loginForm');
    const userIdInput = document.getElementById('userId');
    const userPasswordInput = document.getElementById('userPassword');
    const userNameDisplay = document.getElementById('userName');
    const errorMessage = document.getElementById('errorMessage');
    const successMessage = document.getElementById('successMessage');
    const loading = document.getElementById('loading');
    const loginBtn = document.getElementById('loginBtn');

    // 숫자만 입력 가능하도록 필터링
    let nameCheckTimeout = null;
    
    userIdInput.addEventListener('input', function(e) {
        e.target.value = e.target.value.replace(/[^0-9]/g, '');
        
        // 타이핑 중에는 이전 요청 취소
        if (nameCheckTimeout) {
            clearTimeout(nameCheckTimeout);
        }
        
        // 7-8자리 입력 시 사용자 이름 가져오기
        // 8자리를 우선적으로 확인하기 위해 약간의 지연 추가
        if (e.target.value.length >= 7 && e.target.value.length <= 8) {
            nameCheckTimeout = setTimeout(() => {
                fetchUserName(e.target.value);
            }, 300); // 300ms 지연으로 사용자가 8자리까지 입력할 시간 제공
        } else {
            userNameDisplay.textContent = '';
        }
    });

    // 비밀번호 입력 필터링 (숫자만)
    userPasswordInput.addEventListener('input', function(e) {
        e.target.value = e.target.value.replace(/[^0-9]/g, '');
    });

    // 사용자 이름 가져오기
    async function fetchUserName(userId) {
        try {
            const response = await fetch('/api/get_user_name', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ user_id: userId })
            });

            const data = await response.json();
            
            if (data.success && data.user_name) {
                userNameDisplay.textContent = data.user_name;
                userNameDisplay.style.color = '#27ae60';
            } else {
                userNameDisplay.textContent = '등록되지 않은 사번';
                userNameDisplay.style.color = '#e74c3c';
            }
        } catch (error) {
            console.error('Error fetching user name:', error);
            userNameDisplay.textContent = '';
        }
    }

    // 로그인 폼 제출
    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // 에러 메시지 초기화
        errorMessage.classList.remove('show');
        successMessage.classList.remove('show');
        
        const userId = userIdInput.value;
        const userPassword = userPasswordInput.value;
        
        // 유효성 검사
        if (!userId || userId.length < 7 || userId.length > 8) {
            showError('사번은 7-8자리 숫자여야 합니다.');
            return;
        }
        
        if (!userPassword) {
            showError('비밀번호를 입력해주세요.');
            return;
        }
        
        if (!/^\d+$/.test(userId)) {
            showError('사번은 숫자만 입력 가능합니다.');
            return;
        }
        
        if (!/^\d+$/.test(userPassword)) {
            showError('비밀번호는 숫자만 입력 가능합니다.');
            return;
        }
        
        // 로딩 표시
        loading.classList.add('show');
        loginBtn.disabled = true;
        
        try {
            const response = await fetch('/api/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: userId,
                    user_password: userPassword
                })
            });

            const data = await response.json();
            
            if (data.success) {
                showSuccess('로그인 성공! 잠시 후 메인 페이지로 이동합니다.');
                // 로그인 성공 시 버튼 비활성화 유지
                setTimeout(() => {
                    window.location.href = '/dashboard';
                }, 1500);
            } else {
                showError(data.message || '로그인에 실패했습니다.');
                // 로그인 실패 시에만 버튼 재활성화
                loading.classList.remove('show');
                loginBtn.disabled = false;
            }
        } catch (error) {
            console.error('Login error:', error);
            showError('서버 오류가 발생했습니다. 잠시 후 다시 시도해주세요.');
            // 에러 발생 시에만 버튼 재활성화
            loading.classList.remove('show');
            loginBtn.disabled = false;
        }
    });

    // 에러 메시지 표시
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.add('show');
        setTimeout(() => {
            errorMessage.classList.remove('show');
        }, 5000);
    }

    // 성공 메시지 표시
    function showSuccess(message) {
        successMessage.textContent = message;
        successMessage.classList.add('show');
    }

    // Enter 키로 로그인
    userPasswordInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            loginForm.dispatchEvent(new Event('submit'));
        }
    });
});