const dispatchHandlers = () => {
    document.body.onclick = (e) => {
        if (e.target.tagName === 'A' && e.target.parentNode.parentNode.className === 'menu-list') {
            try {
                document.getElementsByClassName('is-active')[0].classList.remove('is-active');
            } catch(classNotYetUsed) {
                ;
            } finally {
                e.target.classList.add('is-active');
            }
        }
    }
}